# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use
[GitHub pull requests](https://help.github.com/articles/about-pull-requests/)
for this purpose.

1.  When a PR is created, the **Google oncall** triages it and adds a **Google
    domain expert** as its **reviewer**.
1.  The **domain expert** reviews the PR, approves it, and then adds a **Google
    quality champion** as a reviewer AND the **assignee**.
    *   For now the domain expert needs to manually find a quality champion.
        Later we'll try to automate it.
    *   See the image below for how the domain expert and quality champion roles
        map to the GitHub UI.

        <img src="docs/pr-roles.png" alt="PR Roles" width="400px">
1.  The **Google quality champion** reviews the PR, focusing on style and
    quality, approves it, and then applies the `pull ready` label to the PR.
1.  The `pull ready` label triggers the bot to generate a CL from the PR:
    *   The **author** will be the bot.
    *   The **original author** will be the (external) PR author.
    *   The **reviewer** will be the Googler assigned the PR (the quality
        champion), who's responsible for approving the CL
    *   The Google domain expert and all other reviewers will be **CC**-ed.
    *   As long as the quality champion has approved the PR, anyone is allowed
        to apply the `pull ready` label in case the quality champion forgets to
        do so.
1.  By this time, the CL should most likely be ready. If, however, more changes
    are needed (e.g. due to presubmit failures), the Google domain expert is
    usually responsible for making such changes. They can pick one of two ways
    depending on the task:
    1.  **Patch the CL into a workspace**, iterate there, get approval from the
        quality champion, and submit it. Make sure to preserve the CL
        description (which contains the PR link) so that the bot can close the
        original PR automatically.
    1.  **Make changes to the PR's branch** (exact instructions to be
        documented), let the bot update the CL, iterate until the CL is ready,
        and get approval from the quality champion. The CL will auto submit, and
        the original PR will auto close.
1.  In rare cases, the internal testing of the CL may reveal an issue that
    requires a big change to the PR's design. The Google domain expert should
    work with the external contributor to address it, possibly via a new PR.
    *   If more changes are made to the original PR, a Google engineer will need
        to re-approve the PR for the changes to propagate to the CL.

Notes:

*   When a PR is ready for another review, click on the "Re-request review"
    (spinning arrows) button next to a reviewer name to notify them.
*   We want the quality champion's review to be in the open so that external
    contributors can learn Google's coding standard.
*   The bot-generated CL cannot be changed manually, but if you update the PR it
    is connected to, the bot will update it automatically, or the CL can be
    patched into a new workspace controlled by an internal user.
