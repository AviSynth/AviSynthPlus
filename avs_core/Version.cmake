if (EXISTS ${REPO}/.git)
EXECUTE_PROCESS(
    COMMAND "${GIT}" --git-dir=${REPO}/.git  rev-list --count HEAD
    OUTPUT_VARIABLE AVS_SEQREV
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
EXECUTE_PROCESS(
    COMMAND "${GIT}" --git-dir=${REPO}/.git  rev-parse --abbrev-ref HEAD
    OUTPUT_VARIABLE AVS_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# find the newest created tag, which will hopefully only be relevant to
# release tags (which themselves only apply to the release branches)
EXECUTE_PROCESS(
    COMMAND "${GIT}" --git-dir=${REPO}/.git  describe --tags --abbrev=0
    OUTPUT_VARIABLE AVS_NEWEST_TAG
)
string(STRIP ${AVS_NEWEST_TAG} AVS_NEWEST_TAG)

# count the number of commits since the most recently created tag.
# if an older-than-tag commit has been checked out as HEAD, then this
# will report '0', which shouldn't be a problem because this is entirely
# intended for the purposes of current development and not as an
# arbitrary meter between release tags.
EXECUTE_PROCESS(
    COMMAND "${GIT}" --git-dir=${REPO}/.git  rev-list --count ${AVS_NEWEST_TAG}..HEAD
    OUTPUT_VARIABLE AVS_DEVNEXT_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# date of last git commit to branch
EXECUTE_PROCESS(
    COMMAND "${GIT}" --git-dir=${REPO}/.git  log -1 HEAD --format=%cd --date=unix
    OUTPUT_VARIABLE AVS_DEV_REVDATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(ENV{SOURCE_DATE_EPOCH} ${AVS_DEV_REVDATE})
string(TIMESTAMP AVS_DEV_REVDATE %Y-%m-%d UTC)
unset(ENV{SOURCE_DATE_EPOCH})

# abbreviated git commit hash
EXECUTE_PROCESS(
    COMMAND "${GIT}" --git-dir=${REPO}/.git  describe --tags
    OUTPUT_VARIABLE AVS_DEV_GITHASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
endif()
CONFIGURE_FILE(${SRC} ${DST} @ONLY)
