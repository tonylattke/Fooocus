"""Opt-in local smoke test for downloaded video models.

Examples:
  PYTHONPATH=. python tests/smoke_video.py --model wan --image input.png
  PYTHONPATH=. python tests/smoke_video.py --model h3 --image input.png
"""

import argparse
import sys


def main() -> int:
    original_arguments = sys.argv[1:]
    # Fooocus's args_manager parses argv during module import.
    sys.argv = [sys.argv[0]]
    from modules.video_worker import VideoTask, process_task

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["wan", "h3"], default="wan")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="A slow cinematic camera move")
    parser.add_argument("--profile", choices=["Auto", "8 GB", "12 GB", "16 GB+"], default="Auto")
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(original_arguments)

    task = VideoTask(
        image=args.image,
        prompt=args.prompt,
        model=args.model,
        hardware_profile=args.profile,
        duration=args.duration,
        seed=args.seed,
    )
    process_task(task)
    for event, payload in task.yields:
        print(f"{event}: {payload}")
    return 0 if any(event == "finish" for event, _payload in task.yields) else 1


if __name__ == "__main__":
    raise SystemExit(main())
