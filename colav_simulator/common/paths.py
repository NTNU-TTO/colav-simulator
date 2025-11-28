"""
paths.py

Summary:
    Contains paths to default configuration files and schemas.

Author: Trym Tengesdal
"""

import pathlib


def _get_package_root():
    """Get the root directory of the colav_simulator package.

    Returns:
        Path: Package root directory (always the actual package, never project root)

    Raises:
        RuntimeError: If package root cannot be determined
    """
    package_file = pathlib.Path(__file__).absolute()
    package_root = package_file.parents[1]

    # If installed in site-packages, we're good
    if "site-packages" in str(package_root) or "dist-packages" in str(package_root):
        return package_root

    # Otherwise, verify it's local development (project root should have config/scenarios)
    project_root = package_root.parent
    if not (
        (project_root / "config").exists() or (project_root / "scenarios").exists()
    ):
        raise RuntimeError(
            f"Could not determine package root. "
            f"Package file: {package_file}, "
            f"Package root: {package_root}, "
            f"Project root: {project_root}"
        )

    return package_root


package = _get_package_root()
is_installed = "site-packages" in str(package) or "dist-packages" in str(package)
schemas = package / "schemas"
if is_installed:
    # Installed: everything is in the package directory
    config = package / "config"
    scenarios = package / "scenarios"
    output = package / "output"
else:
    # Local development: schemas are in package, config/scenarios are in project root
    project_root = package.parent
    config = project_root / "config"
    scenarios = project_root / "scenarios"
    output = project_root / "output"

simulator_schema = schemas / "simulator.yaml"
scenario_schema = schemas / "scenario.yaml"
scenario_generator_schema = schemas / "scenario_generator.yaml"

simulator_config = config / "simulator.yaml"
scenario_generator_config = config / "scenario_generator.yaml"
seacharts_config = config / "seacharts.yaml"

enc_data = package / "data" / "map"
ais_data = package / "data" / "ais"
saved_scenarios = scenarios / "saved"

animation_output = output / "animations"
figure_output = output / "figures"
