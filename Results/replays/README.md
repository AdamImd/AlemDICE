# Environment Replays

These are environment-only team replays. They do not include LLM prompts,
responses, or per-agent prompt visualizations.

- [100-tick Source vs. embodied-commander comparison](embodied_commander_100_side_by_side.mp4) —
  synchronized 10 FPS full-world replay of paired Easy seed `12000`; Source is
  on the left and the embodied commander is on the right.
- [100-tick Source baseline](embodied_commander_100_source_full_world.mp4) —
  all 100 saved states from the canonical Source arm of
  `20260723T021700Z_embodied_commander_100`.
- [100-tick embodied commander](embodied_commander_100_commander_full_world.mp4) —
  all 100 saved states from the canonical `embodied_commander_broadcast` arm of
  `20260723T021700Z_embodied_commander_100`.
- [Longest N=3 full-world demo (MP4)](n3_longest_514_step_full_world.mp4) —
  51.4-second, 768×832 rendering of all 514 saved states from
  `2026-07-21_01-22-48_robust_all_gemma4-31b_easy/default_run_03`, the longest
  complete state bundle found across the `AlemDICE_dev` and `AlemDICE`
  workspaces.
- [N=3 full-world reconstruction (MP4)](n3_full_world.mp4) — post-hoc 768×832
  rendering of all 200 saved world states, including level transitions and all
  labeled agents.
- [N=3 full-world test video](n3_full_world_test.mp4) — ten sampled states at
  two frames per second; kept separate from the complete production replay.
