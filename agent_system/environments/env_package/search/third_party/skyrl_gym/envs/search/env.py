import re
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import compute_score
from agent_system.environments.env_package.search.third_party.skyrl_gym.tools import SearchToolGroup


class SearchEnv(BaseTextEnv):
    """Environment for Search-R1 style retrieval-augmented QA tasks."""

    def __init__(self, env_config: DictConfig):
        super().__init__()
        self.tool_group = SearchToolGroup(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        self.init_tool_groups([self.tool_group])
        self.search_reward_coef = env_config.search_reward_coef

    def reset(self, extras: Dict[str, Any] = None) -> None:
        extras = extras or {}
        assert "ground_truth" in extras, "ground_truth is required in extras field"
        self.ground_truth = extras["ground_truth"]
        self.max_turns = extras.get("max_turns", 3)
        self.data_source = extras.get("data_source", "unknown")
        self.skill_type = extras.get("skill_type")
        self.chat_history: ConversationType = []
        self.done = False
        self.turns = 0

    def _parse_action(self, action: str) -> List[Optional[str]]:
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def _get_reward(self, action: str, done: bool) -> float:
        if not done:
            return 0.0
        chat_history_str = "".join(item["content"] for item in self.chat_history)
        return compute_score(chat_history_str, self.ground_truth)

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _postprocess_action(self, action: str) -> str:
        if "</search>" in action:
            return action.split("</search>")[0] + "</search>"
        if "</answer>" in action:
            return action.split("</answer>")[0] + "</answer>"
        return action

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> Optional[str]:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)
        if len(tool_output) > 0:
            return "\n<information>" + tool_output + "</information>\n"
        return None

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        if not self.done:
            done = self._is_done(action)
            self.done = done
        else:
            done = True

        reward = self._get_reward(action, done)
        if done:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=done,
                metadata={
                    "data_source": self.data_source,
                    "skill_type": self.skill_type,
                    "tool_calling": False,
                },
                postprocessed_action=action,
            )

        error = None
        query = [None]
        try:
            query = self._parse_action(action)
            if query[0]:
                reward += self.search_reward_coef
            observation = self._execute_tool("SearchToolGroup", "search", query)
        except Exception as exc:
            error = str(exc)
            observation = None

        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            print(f"!!(Warning) an error when calling tools: {error}")
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_calling": True,
            "tool_group": "SearchToolGroup",
            "tool_name": "search",
            "tool_input": query,
            "data_source": self.data_source,
            "skill_type": self.skill_type,
        }
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
            postprocessed_action=action,
        )
