# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import time
import signal
import syslog


class Timer(object):
    """
    Simple Timer object.
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type_, value, traceback):
        if self.name:
            print("[%s]" % self.name, end=" ")
        print("Elapsed: %s" % (time.time() - self.tstart))


def syslog_elapsed_time(seconds, msg):
    """
    Log a message (msg) to syslog with elapsed time (seconds).
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    syslog.syslog("%s Elapsed: %d:%02d:%02d" % (msg, h, m, s))


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print("\nSIGINT received. Delaying KeyboardInterrupt.", flush=True)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def test():
    with Timer("test"):
        time.sleep(0.1)

    syslog_elapsed_time(1234, "Hello.")

    with DelayedKeyboardInterrupt():
        time.sleep(100)


if __name__ == "__main__":
    test()
