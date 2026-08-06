"""
Split claude_style_skills.json into ID and OOD skill files for the OOD experiment.

ID domains (training): pick_and_place (type 1), clean (type 3), cool (type 5)
OOD domains (testing): look_at_obj_in_light (type 2), heat (type 4), pick_two_obj (type 6), examine (type 2)

Usage:
    python scripts/split_skills_ood.py
"""
import json
import os
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(PROJECT_ROOT, "memory_data/alfworld/claude_style_skills.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "memory_data/alfworld_ood")
ID_OUTPUT = os.path.join(OUTPUT_DIR, "claude_style_skills_id.json")
OOD_OUTPUT = os.path.join(OUTPUT_DIR, "claude_style_skills_ood.json")

# Skill categories belonging to each split
ID_CATEGORIES = ["pick_and_place", "clean", "cool"]
OOD_CATEGORIES = ["look_at_obj_in_light", "examine", "heat"]  # existing ones

# New pick_two_obj skills for type 6 — designed from trajectory analysis across
# early (sr=0.19), mid (sr=0.36), and late (sr=0.33) training phases.
# Key failure modes observed:
#   1. Forgetting to place the 2nd object after placing the 1st (premature termination)
#   2. Re-visiting already-empty locations when searching for the 2nd object
#   3. Picking up an object then putting it back down or moving it to the wrong place
#   4. Action loops: repeating the same go-to/examine cycle for 50+ steps
#   5. Confusing object IDs (e.g., taking saltshaker 1, dropping it, taking saltshaker 3)
PICK_TWO_OBJ_SKILLS = [
    {
        "skill_id": "pt2_001",
        "title": "Two-Phase Workflow",
        "principle": "Treat every pick-two task as two sequential pick-and-place sub-tasks. Phase 1: find object #1, carry it to the target, place it. Phase 2: find object #2, carry it to the same target, place it. Never start Phase 2 until Phase 1's placement is confirmed by the environment feedback.",
        "when_to_apply": "Immediately after reading a 'put two X in Y' goal; use this as the top-level plan structure."
    },
    {
        "skill_id": "pt2_002",
        "title": "Never Drop a Held Goal Object",
        "principle": "Once you pick up a goal object, never put it down anywhere except the final target location. Dropping it on a random surface or placing it in the wrong receptacle wastes steps and risks losing track of it. If you realize you are at the wrong location, navigate to the correct target while still holding the object.",
        "when_to_apply": "Any time you are holding an object that matches the goal and are about to issue a 'put' or 'move' action — verify the destination matches the goal before executing."
    },
    {
        "skill_id": "pt2_003",
        "title": "Remember the Target Location",
        "principle": "The target receptacle (e.g., 'drawer', 'shelf', 'cabinet') is the same for both objects. After placing the first object, remember its exact ID (e.g., 'drawer 1') so you can navigate directly back for the second placement without re-searching for the destination.",
        "when_to_apply": "After successfully placing the first object: record the target ID. When carrying the second object: go directly to the recorded target."
    },
    {
        "skill_id": "pt2_004",
        "title": "Partition Search Space Between Phases",
        "principle": "During Phase 1, mentally log every location visited and whether it contained a matching object. When entering Phase 2, skip all locations confirmed empty in Phase 1 and only search new or previously closed containers. This prevents the common failure of looping over the same empty cabinets for 40+ steps.",
        "when_to_apply": "When transitioning from Phase 1 (first object placed) to Phase 2 (searching for second object)."
    },
    {
        "skill_id": "pt2_005",
        "title": "Strict Completion Check",
        "principle": "The task is only complete when exactly two objects have been confirmed placed at the target by environment feedback (e.g., 'You put the X in/on the Y'). After placing the first object, you MUST continue — do not issue 'done' or stop acting. Count: placed=0 → search, placed=1 → search again, placed=2 → stop.",
        "when_to_apply": "After every successful placement action: increment the placed counter and decide whether to continue searching or terminate."
    },
    {
        "skill_id": "pt2_006",
        "title": "Break Action Loops Aggressively",
        "principle": "If you find yourself visiting the same location or executing the same action more than twice without progress, immediately switch to a completely different area of the room. Common loop traps: alternating between two surfaces, repeatedly opening the same container, going back and forth between the target and a single search spot.",
        "when_to_apply": "When the last 3-4 actions have produced 'Nothing happens' or revisited locations you already know are empty."
    }
]


def main():
    with open(INPUT_PATH, "r") as f:
        skills = json.load(f)

    general = skills["general_skills"]
    task_specific = skills["task_specific_skills"]
    common_mistakes = skills["common_mistakes"]

    # --- ID skill file ---
    id_task_specific = {cat: task_specific[cat] for cat in ID_CATEGORIES}
    id_skills = {
        "general_skills": copy.deepcopy(general),
        "task_specific_skills": id_task_specific,
        "common_mistakes": copy.deepcopy(common_mistakes),
        "metadata": {
            "source": "split from claude_style_skills.json for OOD experiment (ID split)",
            "id_categories": ID_CATEGORIES,
            "total_general": len(general),
            "total_task_specific": sum(len(v) for v in id_task_specific.values()),
            "total_common_mistakes": len(common_mistakes),
        }
    }

    # --- OOD skill file ---
    ood_task_specific = {cat: task_specific[cat] for cat in OOD_CATEGORIES}
    ood_task_specific["pick_two_obj"] = PICK_TWO_OBJ_SKILLS
    ood_skills = {
        "general_skills": copy.deepcopy(general),
        "task_specific_skills": ood_task_specific,
        "common_mistakes": copy.deepcopy(common_mistakes),
        "metadata": {
            "source": "split from claude_style_skills.json for OOD experiment (OOD split)",
            "ood_categories": OOD_CATEGORIES + ["pick_two_obj"],
            "total_general": len(general),
            "total_task_specific": sum(len(v) for v in ood_task_specific.values()),
            "total_common_mistakes": len(common_mistakes),
        }
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(ID_OUTPUT, "w") as f:
        json.dump(id_skills, f, indent=2, ensure_ascii=False)
    print(f"ID skills written to {ID_OUTPUT}")
    print(f"  general: {len(general)}, task_specific categories: {list(id_task_specific.keys())}, "
          f"task_specific total: {sum(len(v) for v in id_task_specific.values())}, "
          f"common_mistakes: {len(common_mistakes)}")

    with open(OOD_OUTPUT, "w") as f:
        json.dump(ood_skills, f, indent=2, ensure_ascii=False)
    print(f"OOD skills written to {OOD_OUTPUT}")
    print(f"  general: {len(general)}, task_specific categories: {list(ood_task_specific.keys())}, "
          f"task_specific total: {sum(len(v) for v in ood_task_specific.values())}, "
          f"common_mistakes: {len(common_mistakes)}")


if __name__ == "__main__":
    main()
