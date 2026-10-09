import gzip
import os
from pathlib import Path
import tarfile
import tempfile

from setuptools import setup
from setuptools.command.sdist import sdist

os.environ.setdefault("SOURCE_DATE_EPOCH", "315532800")


class AnonymousSdist(sdist):
    """Keep filesystem ownership and creation times out of source archives."""

    def make_archive(self, base_name, format, **kwargs):
        filename = super().make_archive(base_name, format, **kwargs)
        if self.dry_run or format != "gztar":
            return filename
        archive = Path(filename)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as output:
                temporary = Path(output.name)
                with (
                    tarfile.open(archive, "r:gz") as source,
                    gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed,
                    tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target,
                ):
                    for member in source:
                        member.uid = member.gid = member.mtime = 0
                        member.uname = member.gname = ""
                        member.pax_headers = {}
                        content = source.extractfile(member) if member.isfile() else None
                        try:
                            target.addfile(member, content)
                        finally:
                            if content is not None:
                                content.close()
            os.replace(temporary, archive)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return filename


setup(cmdclass={"sdist": AnonymousSdist})
