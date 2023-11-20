# Team Software Development Process (SDP)

## Principles

- We respond to all requests within a single business day.
- We use Github Issues to track our progress and indicate how far along each step is.
- The backlog will have work for every role for at least two additional weeks.
- Every new feature needs to be developed in its own separate branch which will be merged into main.
- Code review has to take place before any pull request is merged - submitter of the request must designate a reviewer who understands the modification.
- Each work item should be very well-defined and take no more than 8 hours of work to complete.
- Changes that alter how our code interacts with the other teams’ work must be discussed with the leads of both teams and clearly noted in the pull request.
- Each major change should be discussed with the whole team and mentor during a meeting.
- Developers of new features should add the appropriate documentation once their feature is approved and merged (preferably should be drafting documentation while writing code).

## Process

- Backlog and Planning in meeting (weekly)
- Kanban-type tracker in Github issues (to do, in progress, done)
- Demo with mentors (weekly)
- Synchronization meeting with other subteams (every other week)
- Communicate over Discord when there are problems that need to be discussed
- Meet with other team leads and TA every week

## Roles

- **Team lead:** Jack Stevenson
  - Duties: Logistics with other subteams, running meetings, coding
- **Software engineering:** Nigel Higgs, Ramish Mohammad, Dillon Baldwin
  - Duties: Researching approaches for problems, writing code, reviewing code changes, analyzing current methodology

## Tooling

- **Version Control:** GitHub
- **Project Management:** GitHub Issues and Projects
- **Documentation:** [Starlight GitHub Repository](https://github.com/withastro/starlight)
- **Test Framework:** TBD
- **Linting and Formatting:** TBD
- **CI/CD:** Azure, Github Actions
- **IDE:** VS Code / personal preference of team members
- **Graphic Design:** Figma
- **Others:** Google Docs

## Definition of Done (DoD)

- Code is correctly formatted and commented
- Pull review is submitted and merged into main branch after passing review
- Unit and integration tests running on Github Actions all pass
- All functionality that wasn’t represented in a test (i.e. UI element) still works as designed
- Documentation is updated
- Release notes are updated
- Demo notes are prepared for the next meeting
- Any appropriate information is added to shared change logs for reference by the other subteams

## Release Cycle

- Automatically deploy to staging every merge to main branch
- Deploy to production every release
- Release whenever appropriate - may be a short cycle during initial development
- Use semantic versioning MAJOR.minor.patch
- Increment the minor version for new features
- Increment the patch version for bug fixes
- Increment the major version for breaking API changes

## Environments

| Environment | Infrastructure           | Deployment | What is it for?                                        | Monitoring                   |
|-------------|---------------------------|------------|--------------------------------------------------------|------------------------------|
| Production  | Azure through CI/CD       | Release    | Major features that other subteams will be using      | Prometheus, Grafana, Sentry  |
| Staging     | Azure through CI/CD       | PR         | New unreleased features and integration tests         | Sentry                       |
| Dev         | Local (macOS and Windows) or on Azure dev cloud | Commit | Development and unit tests                             | N/A                          |
