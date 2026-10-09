"""Console progress bar.

Original idea: https://stackoverflow.com/a/15860757/1391441
"""


class ProgressBar:
    """Prints a bar that only redraws when the shown percentage changes."""

    def __init__(self, total, width=20):
        self.total = max(total, 1)
        self.width = width
        self.shown = None

    def update(self, progress):
        fraction = min(max(progress / self.total, 0.), 1.)
        percent = round(100 * fraction)
        if percent == self.shown:
            return
        self.shown = percent
        block = round(self.width * fraction)
        print(f"\r  [{'#' * block}{'-' * (self.width - block)}] {percent}% ",
              end='', flush=True)

    def finish(self):
        self.update(self.total)
        print()
