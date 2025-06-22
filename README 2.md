# flux.template.service

## Initialising your repository

This template repo comes with a GHA workflow that takes care of the standard repository settings. This documentation (with screenshots) will also available on [Confluence](#) in due course.

To use the workflow:

1. Create an ephemeral personal access token with the following permissions:
   - `repo`
   - `write:org`
2. Be sure to configure SSO for the token (click `Configure SSO` and then authorize Sage)
3. Add the access token as a repository secret named `INITIALISE_REPO_PAT`
4. Under Actions, dispatch the `Initialise Repository` workflow

This should do the following:

- Create environments `dev1`, `qa1`, `sbx`, `prod`, with a protection rule from `qa1` upward
- Set generic defaults, such as the default branch (`main`), PR policies, etc
- Add branch protection for `main`
- Add the flux teams, with appropriate permissions

On success:

1. Revoke the PAT
2. Delete the `init-repo.yaml` workflow
3. Delete this section from your repo's README and update the title to match your repo name

On fail:

1. Hope for the best 🥹

## Github Actions Variables and Secrets

The pipelines assume that the following variables/secrets will be accessible:

| NAME                      | VAR/SECRET | REPO/ENV |
| ------------------------- | ---------- | -------- |
| AWS_ACCOUNT_ID            | VAR        | ENV      |
| AWS_REGION                | VAR        | ENV      |
| AWS_CICD_ROLE_NAME        | VAR        | EITHER   |
| AWS_ROLE_DURATION_SECONDS | VAR        | EITHER   |
| SERVICE_NOW_USERNAME      | VAR        | REPO     |
| SERVICE_NOW_PASSWORD      | SECRET     | REPO     |
| SERVICE_NOW_INSTANCE      | VAR        | REPO     |
| IS_CAB_REQUIRED           | VAR        | REPO     |

### Info

- `IS_CAB_REQUIRED` relates to whether or not a release requires CAB approval before going to production. See the [release documentation][1] for further information on this.
- `SERVICE_NOW` related variables will be set by the Flux Devops team

## Contact

For issues and/or questions about this template repo, contact the Flux Devops team:

- [Joao Cerqueira - FuelbigData ](mailto:joao@fuelbigdata.com), Engineer

[1]: https://confluence.sage.com/display/CODP/Release+Process
