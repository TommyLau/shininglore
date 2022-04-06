# shinginglore [![Build Status](https://github.com/TommyLau/shininglore/actions/workflows/ci.yml/badge.svg)](https://github.com/TommyLau/shininglore/actions/workflows/ci.yml)

## Check out the source code with LFS

First, check out the `source code` without pulling LFS files:

```
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:TommyLau/shininglore.git
```

Set the LFS url to Tencent Code:

```
cd shininglore
git config lfs.url git@git.code.tencent.com:ShiningLore/assets.git
git config lfs.https://git.code.tencent.com/ShiningLore/assets.git.locksverify true
```

Then pull the LFS files:

```
git lfs pull
```

Later we can just use `git push` and `git pull` normally.
