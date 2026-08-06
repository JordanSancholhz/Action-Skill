from gymnasium.envs.registration import register

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)