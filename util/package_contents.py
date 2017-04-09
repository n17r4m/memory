# http://stackoverflow.com/questions/487971/is-there-a-standard-way-to-list-names-of-python-modules-in-a-package

import imp
import os
MODULE_EXTENSIONS = ('.py', '.pyc', '.pyo')



def package_contents(package_name):
  file, pathname, description = imp.find_module(package_name)
  if file:
    raise ImportError('Not a package: %r', package_name)
  return list(set([os.path.splitext(module)[0]
    for module in os.listdir(pathname)
    if not module.startswith("__init") and module.endswith(MODULE_EXTENSIONS)]))
