#define _GNU_SOURCE

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define REAL_BWRAP "/usr/libexec/ara-cell/bwrap.real"

static void fail(const char *message) {
    fprintf(stderr, "ara-bwrap-guard: %s\n", message);
    _exit(126);
}

static bool option_is(const char *argument, const char *option) {
    const size_t length = strlen(option);
    return strcmp(argument, option) == 0 ||
           (strncmp(argument, option, length) == 0 && argument[length] == '=');
}

static void reject_outer_proc_path(const char *argument) {
    char *resolved = realpath(argument, NULL);
    if (
        resolved != NULL &&
        (strcmp(resolved, "/proc") == 0 ||
         strncmp(resolved, "/proc/", 6) == 0)
    ) {
        free(resolved);
        fail("binding the outer procfs is forbidden");
    }
    free(resolved);
}

static bool safe_device(const char *path) {
    return strcmp(path, "/dev/null") == 0 ||
           strcmp(path, "/dev/zero") == 0 ||
           strcmp(path, "/dev/random") == 0 ||
           strcmp(path, "/dev/urandom") == 0;
}

static bool bind_option(const char *argument) {
    return strcmp(argument, "--bind") == 0 ||
           strcmp(argument, "--bind-try") == 0 ||
           strcmp(argument, "--dev-bind") == 0 ||
           strcmp(argument, "--dev-bind-try") == 0 ||
           strcmp(argument, "--ro-bind") == 0 ||
           strcmp(argument, "--ro-bind-try") == 0;
}

static bool exact_pair(
    const char *source,
    const char *destination,
    const char *expected_source,
    const char *expected_destination
) {
    return strcmp(source, expected_source) == 0 &&
           strcmp(destination, expected_destination) == 0;
}

static bool approved_readonly_pair(
    const char *source,
    const char *destination
) {
    return exact_pair(source, destination, "/", "/") ||
           exact_pair(
               source,
               destination,
               "/usr/bin/python3.10",
               "/usr/bin/python3.10"
           ) ||
           exact_pair(
               source,
               destination,
               "/usr/bin/setpriv",
               "/usr/bin/setpriv"
           ) ||
           exact_pair(
               source,
               destination,
               "/usr/lib/python3.10",
               "/usr/lib/python3.10"
           ) ||
           exact_pair(
               source,
               destination,
               "/usr/lib/x86_64-linux-gnu",
               "/usr/lib/x86_64-linux-gnu"
           ) ||
           exact_pair(
               source,
               destination,
               "/lib/x86_64-linux-gnu",
               "/lib/x86_64-linux-gnu"
           ) ||
           exact_pair(
               source,
               destination,
               "/lib64/ld-linux-x86-64.so.2",
               "/lib64/ld-linux-x86-64.so.2"
           ) ||
           exact_pair(
               source,
               destination,
               "/opt/ara_gsplat/runtime/controller_worker.py",
               "/sandbox/controller_worker.py"
           );
}

static void require_operands(
    int argc,
    char **argv,
    int option_index,
    int count
) {
    if (option_index + count >= argc) {
        fail("setup option is missing a required operand");
    }
    for (int offset = 1; offset <= count; ++offset) {
        if (strcmp(argv[option_index + offset], "--") == 0) {
            fail("setup option is missing a required operand");
        }
    }
}

static bool decimal_fd(const char *text) {
    if (text[0] == '\0') {
        return false;
    }
    for (const char *cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            return false;
        }
    }
    return true;
}

static bool approved_dir(const char *path) {
    return strcmp(path, "/usr") == 0 ||
           strcmp(path, "/usr/bin") == 0 ||
           strcmp(path, "/usr/lib") == 0 ||
           strcmp(path, "/lib") == 0 ||
           strcmp(path, "/lib64") == 0 ||
           strcmp(path, "/dev") == 0 ||
           strcmp(path, "/sandbox") == 0;
}

static bool approved_tmpfs(const char *path) {
    return strcmp(path, "/tmp") == 0 ||
           strcmp(path, "/app/.git") == 0 ||
           strcmp(path, "/app/.agents") == 0 ||
           strcmp(path, "/app/.codex") == 0 ||
           strcmp(path, "/tmp/.git") == 0 ||
           strcmp(path, "/tmp/.agents") == 0 ||
           strcmp(path, "/tmp/.codex") == 0;
}

static bool approved_readonly_remount(const char *path) {
    return strcmp(path, "/app/.git") == 0 ||
           strcmp(path, "/app/.agents") == 0 ||
           strcmp(path, "/app/.codex") == 0 ||
           strcmp(path, "/tmp/.git") == 0 ||
           strcmp(path, "/tmp/.agents") == 0 ||
           strcmp(path, "/tmp/.codex") == 0;
}

static void validate_bind(
    const char *option,
    const char *source,
    const char *destination
) {
    if (source[0] != '/' || destination[0] != '/') {
        fail("bind sources and destinations must be absolute paths");
    }
    if (
        strcmp(option, "--bind-try") == 0 ||
        strcmp(option, "--dev-bind-try") == 0 ||
        strcmp(option, "--ro-bind-try") == 0
    ) {
        fail("best-effort bind options are forbidden");
    }
    if (strcmp(option, "--bind") == 0) {
        const bool approved_pair =
            (strcmp(source, "/app") == 0 &&
             strcmp(destination, "/app") == 0) ||
            (strcmp(source, "/tmp") == 0 &&
             strcmp(destination, "/tmp") == 0);
        if (!approved_pair) {
            fail("writable binds are limited to exact /app and /tmp pairs");
        }
    }
    if (strcmp(option, "--dev-bind") == 0) {
        if (
            !safe_device(source) ||
            strcmp(source, destination) != 0
        ) {
            fail("device binds are limited to exact safe pseudo-device pairs");
        }
    }
    if (
        strcmp(option, "--ro-bind") == 0 &&
        !approved_readonly_pair(source, destination)
    ) {
        fail("read-only bind pair is outside the fixed sandbox allowlist");
    }
    char *resolved = realpath(source, NULL);
    if (resolved == NULL) {
        fail("bind source must exist before sandbox construction");
    }
    if (
        strcmp(resolved, "/proc") == 0 ||
        strncmp(resolved, "/proc/", 6) == 0
    ) {
        free(resolved);
        fail("binding the outer procfs is forbidden");
    }
    if (strcmp(resolved, "/") == 0) {
        if (
            strcmp(option, "--ro-bind") != 0 ||
            strcmp(source, "/") != 0 ||
            strcmp(destination, "/") != 0
        ) {
            free(resolved);
            fail("outer root must be one literal read-only root mount");
        }
    }
    if (
        strcmp(resolved, "/dev") == 0 ||
        strncmp(resolved, "/dev/", 5) == 0
    ) {
        const bool device_bind =
            strcmp(option, "--dev-bind") == 0 ||
            strcmp(option, "--dev-bind-try") == 0;
        if (
            !device_bind ||
            !safe_device(resolved) ||
            strcmp(source, resolved) != 0 ||
            strcmp(destination, resolved) != 0
        ) {
            free(resolved);
            fail("only exact safe pseudo-device binds are permitted");
        }
    }
    reject_outer_proc_path(destination);
    free(resolved);
}

int main(int argc, char **argv) {
    bool unshare_pid = false;
    bool unshare_net = false;
    bool die_with_parent = false;
    bool new_session = false;
    int separator = -1;
    char **guarded = NULL;
    int output = 0;

    if (
        argc == 2 &&
        (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "--help") == 0)
    ) {
        if (setgid(getgid()) != 0 || setuid(getuid()) != 0) {
            fail("could not drop privileges for informational mode");
        }
        execv(REAL_BWRAP, argv);
        fail("could not execute informational mode");
    }
    if (geteuid() != 0) {
        fail("setuid root is unavailable");
    }

    /*
     * Parse and construct the trusted argv in one pass. In particular, an
     * option-looking operand is still an operand: it must never satisfy one
     * of the required isolation flags. Unknown setup options fail closed so
     * their arity cannot desynchronise this parser from Bubblewrap's parser.
     */
    guarded = calloc((size_t)argc + 5U, sizeof(*guarded));
    if (guarded == NULL) {
        fail("allocation failed");
    }
    guarded[output++] = (char *)REAL_BWRAP;

    for (int index = 1; index < argc; ++index) {
        const char *arg = argv[index];
        if (strcmp(arg, "--") == 0) {
            separator = index;
            break;
        }
        if (bind_option(arg)) {
            require_operands(argc, argv, index, 2);
            validate_bind(arg, argv[index + 1], argv[index + 2]);
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            guarded[output++] = argv[index + 2];
            index += 2;
        } else if (
            option_is(arg, "--bind") ||
            option_is(arg, "--bind-try") ||
            option_is(arg, "--dev-bind") ||
            option_is(arg, "--dev-bind-try") ||
            option_is(arg, "--ro-bind") ||
            option_is(arg, "--ro-bind-try")
        ) {
            fail("inline bind option forms are forbidden");
        } else if (strcmp(arg, "--unshare-pid") == 0) {
            unshare_pid = true;
            guarded[output++] = argv[index];
        } else if (strcmp(arg, "--unshare-net") == 0) {
            unshare_net = true;
            guarded[output++] = argv[index];
        } else if (strcmp(arg, "--die-with-parent") == 0) {
            die_with_parent = true;
            guarded[output++] = argv[index];
        } else if (strcmp(arg, "--new-session") == 0) {
            new_session = true;
            guarded[output++] = argv[index];
        } else if (
            strcmp(arg, "--unshare-ipc") == 0 ||
            strcmp(arg, "--unshare-uts") == 0 ||
            strcmp(arg, "--clearenv") == 0
        ) {
            guarded[output++] = argv[index];
        } else if (
            strcmp(arg, "--unshare-user") == 0 ||
            strcmp(arg, "--unshare-user-try") == 0
        ) {
            /* The setuid guard supplies privilege; nested userns may be blocked. */
            continue;
        } else if (
            strcmp(arg, "--share-pid") == 0 ||
            strcmp(arg, "--share-net") == 0
        ) {
            fail("share-pid/share-net is forbidden");
        } else if (
            strcmp(arg, "--unshare-all") == 0 ||
            option_is(arg, "--userns") ||
            option_is(arg, "--userns2") ||
            option_is(arg, "--pidns") ||
            option_is(arg, "--uid") ||
            option_is(arg, "--gid") ||
            option_is(arg, "--cap-add") ||
            option_is(arg, "--bind-fd") ||
            option_is(arg, "--ro-bind-fd")
        ) {
            fail("namespace, privilege, and opaque bind-fd options are forbidden");
        } else if (
            strcmp(arg, "--args") == 0 ||
            strncmp(arg, "--args=", 7) == 0
        ) {
            fail("argument-file expansion is forbidden");
        } else if (strcmp(arg, "--proc") == 0) {
            require_operands(argc, argv, index, 1);
            if (strcmp(argv[index + 1], "/proc") != 0) {
                fail("--proc must target exactly /proc");
            }
            ++index;
        } else if (strcmp(arg, "--dev") == 0) {
            require_operands(argc, argv, index, 1);
            if (strcmp(argv[index + 1], "/dev") != 0) {
                fail("--dev must target exactly /dev");
            }
            ++index;
        } else if (option_is(arg, "--proc") || option_is(arg, "--dev")) {
            fail("inline proc and dev option forms are forbidden");
        } else if (strcmp(arg, "--hostname") == 0) {
            require_operands(argc, argv, index, 1);
            if (strcmp(argv[index + 1], "ara-visible-controller") != 0) {
                fail("hostname is outside the fixed sandbox allowlist");
            }
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            ++index;
        } else if (strcmp(arg, "--dir") == 0) {
            require_operands(argc, argv, index, 1);
            if (!approved_dir(argv[index + 1])) {
                fail("directory is outside the fixed sandbox allowlist");
            }
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            ++index;
        } else if (strcmp(arg, "--tmpfs") == 0) {
            require_operands(argc, argv, index, 1);
            if (!approved_tmpfs(argv[index + 1])) {
                fail("tmpfs is outside the fixed sandbox allowlist");
            }
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            ++index;
        } else if (strcmp(arg, "--remount-ro") == 0) {
            require_operands(argc, argv, index, 1);
            if (!approved_readonly_remount(argv[index + 1])) {
                fail("read-only remount is outside the fixed sandbox allowlist");
            }
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            ++index;
        } else if (strcmp(arg, "--chdir") == 0) {
            require_operands(argc, argv, index, 1);
            if (strcmp(argv[index + 1], "/sandbox") != 0) {
                fail("working directory is outside the fixed sandbox allowlist");
            }
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            ++index;
        } else if (strcmp(arg, "--perms") == 0) {
            require_operands(argc, argv, index, 1);
            if (
                strcmp(argv[index + 1], "555") != 0 &&
                strcmp(argv[index + 1], "0400") != 0
            ) {
                fail("mount permissions are outside the fixed sandbox allowlist");
            }
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            ++index;
        } else if (strcmp(arg, "--cap-drop") == 0) {
            require_operands(argc, argv, index, 1);
            if (strcmp(argv[index + 1], "ALL") != 0) {
                fail("only dropping all capabilities is permitted");
            }
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            ++index;
        } else if (strcmp(arg, "--setenv") == 0) {
            require_operands(argc, argv, index, 2);
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            guarded[output++] = argv[index + 2];
            index += 2;
        } else if (strcmp(arg, "--ro-bind-data") == 0) {
            require_operands(argc, argv, index, 2);
            if (
                !decimal_fd(argv[index + 1]) ||
                strcmp(argv[index + 2], "/sandbox/controller.py") != 0
            ) {
                fail("read-only data bind is outside the fixed sandbox allowlist");
            }
            guarded[output++] = argv[index];
            guarded[output++] = argv[index + 1];
            guarded[output++] = argv[index + 2];
            index += 2;
        } else {
            fail("setup option is outside the fixed sandbox allowlist");
        }
    }

    if (separator < 0 || separator + 1 >= argc) {
        fail("an explicit -- command separator is required");
    }
    if (!unshare_pid || !unshare_net || !die_with_parent || !new_session) {
        fail("pid, network, parent-death, and session isolation are required");
    }

    /*
     * A read-only root bind can recursively carry the parent's /dev. Mount a
     * fresh device filesystem after every caller-supplied mount so no ordering
     * trick can reveal NVIDIA or other outer device nodes.
     */
    guarded[output++] = "--dev";
    guarded[output++] = "/dev";
    guarded[output++] = "--proc";
    guarded[output++] = "/proc";
    guarded[output++] = "--";
    for (int index = separator + 1; index < argc; ++index) {
        guarded[output++] = argv[index];
    }
    guarded[output] = NULL;

    execv(REAL_BWRAP, guarded);
    fprintf(stderr, "ara-bwrap-guard: exec %s failed: %s\n",
            REAL_BWRAP, strerror(errno));
    return 126;
}
