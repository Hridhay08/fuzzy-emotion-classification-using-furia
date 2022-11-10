import weka.core.packages as packages
import weka.core.jvm as jvm

FURIA_PACKAGE_NAME = "fuzzyUnorderedRuleInduction"
FURIA_PACKAGE_PATH = "fuzzyUnorderedRuleInduction1.0.2.zip"

jvm.start(system_cp=True, packages=True)
items = packages.installed_packages()
package_found = False
for item in items:
    package_name = item.name
    if package_name == FURIA_PACKAGE_NAME:
        package_found = True
        break
if package_found == False:
    packages.install_package(FURIA_PACKAGE_PATH)

jvm.stop()