from nox_poetry import Session, session


@session()
def tests(session: Session) -> None:
    args = session.posargs or ["--cov", "--cov-report=xml"]
    session.install(".[all]")
    session.install(
        "torch-scatter", "-f", "https://data.pyg.org/whl/torch-2.0.0+cpu.html"
    )
    session.install(
        "torch-sparse", "-f", "https://data.pyg.org/whl/torch-2.0.0+cpu.html"
    )
    session.install("strawman")
    session.install("pytest")
    session.install("pytest-cov")
    session.install("pytest-mock")
    session.run("pytest", *args)


locations = ["src", "tests", "noxfile.py"]


@session()
def lint(session: Session) -> None:
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--hook-stage=manual",
        *session.posargs,
    )


@session()
def style_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", "check", *args)


@session()
def pyroma(session: Session) -> None:
    session.install("poetry-core>=1.0.0")
    session.install("pyroma")
    session.run("pyroma", "--min", "10", ".")


@session()
def type_checking(session: Session) -> None:
    args = session.posargs or locations
    session.install(".[all]")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        *args,
    )


@session()
def doctests(session: Session) -> None:
    session.install(".[all]")
    session.install("xdoctest")
    session.install("pygments")
    session.run("xdoctest", "-m", "klinker")


@session()
def build_docs(session: Session) -> None:
    session.install(".[docs]")
    session.run("mkdocs", "build", external=True)
