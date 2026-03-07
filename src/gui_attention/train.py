"""Compatibility wrapper for the SFT training entrypoint."""


def __getattr__(name):
    from gui_attention.training import sft as _sft

    return getattr(_sft, name)


if __name__ == "__main__":
    from gui_attention.training.sft import main

    main()
