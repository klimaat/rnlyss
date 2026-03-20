# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import time
import signal
import syslog
import tqdm
from os import utime
from email.utils import parsedate_tz, mktime_tz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


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


def create_session(**kwargs):
    # Setup a Session with retries
    retry_strategy = Retry(**kwargs)
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def download_file(url, dst, params=None, session=None):
    # Initiate request
    if session is None:
        resp = requests.get(url, params=params, stream=True)
    else:
        resp = session.get(url, params=params, stream=True)

    dst = Path(dst)

    # Check response
    if not resp.ok:
        print(f"{dst} unavailable ...skipping")
        return 0

    # Check if we can use cached value
    try:
        last_modified = mktime_tz(parsedate_tz(resp.headers["Last-Modified"]))
    except KeyError:
        last_modified = 0
    content_length = int(resp.headers.get("Content-Length", 0))

    if dst.is_file():
        stat = dst.stat()
        # If .gz, can't compare equal size; check time
        if dst.suffix == ".gz":
            if int(stat.st_mtime) == last_modified:
                print(f"{dst} unchanged... skipping")
                return stat.st_size

        # Otherwise, check size and time
        if stat.st_size == content_length:
            if int(stat.st_mtime) == last_modified:
                print(f"{dst} unchanged... skipping")
                return content_length

    # Ensure dst directory exists
    dst.parent.mkdir(exist_ok=True)

    # Download with progress bar
    try:
        if dst.suffix == ".gz":
            with gzip.open(dst, "wb") as f, tqdm(
                desc=str(dst.name),
                total=content_length,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024 * 1024):
                    size = f.write(data)
                    bar.update(size)

        else:
            with open(dst, "wb") as f, tqdm(
                desc=str(dst.name),
                total=content_length,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024 * 1024):
                    size = f.write(data)
                    bar.update(size)
    except BaseException:
        # Problem delete file
        if dst.is_file():
            print(f"\n\n{dst} interrupted... deleting\n\n")
            dst.unlink()
            raise

    # Set modification date
    if dst.is_file() and (last_modified > 0):
        utime(dst, (last_modified, last_modified))

    # Return actual size
    return dst.stat().st_size


if __name__ == "__main__":
    test()
