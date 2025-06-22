# Flux Template Service

You need nodejs (UI and CDK) and python3 (API and other LLM-related lambdas) installed on your machine.

> Suggestions: to install Node.js, you can use [nvm (Node Version Manager) ](https://github.com/nvm-sh/nvm) and to install python3, you can use [venv](https://docs.python.org/3/library/venv.html)

## How to deploy the project

This project is deployed on AWS using [AWS CDK](https://docs.aws.amazon.com/cdk/latest/guide/home.html).

To deploy the project, you need to install the AWS CDK CLI and configure your AWS credentials.

Then, you can run the following commands:

```bash
cd infra
npm run cdk:deploy:dev
```
