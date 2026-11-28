/* This consumer deliberately does not include the production ABI header. */
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
  /* Fixed historical v1 symbol, independent of the export and loader macros. */
  void *entry = dlsym(lib, "open" "viking" "_cache_provider_v1");
  void *implementation = dlsym(lib, "cache_provider_v1");
  int result = !entry || !implementation || entry != implementation;
  if (result)
    fprintf(stderr, "missing or incompatible historical v1 entry\n");
  dlclose(lib);
  return result;
}
