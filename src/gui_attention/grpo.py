"""Compatibility wrapper for the GRPO training entrypoint."""


def __getattr__(name):
    from gui_attention.training import grpo as _grpo

    return getattr(_grpo, name)


if __name__ == "__main__":
    from gui_attention.training.grpo import main

    main()
