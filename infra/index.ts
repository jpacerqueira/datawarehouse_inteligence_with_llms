import {
  ApplyFluxStandardTags,
  FluxSolutionsChecks,
} from "@sage/flux-cdk-lib/aspects";
import { getCloudflareCidrRanges } from "@sage/flux-cdk-lib/config";
import { App, Aspects } from "aws-cdk-lib";
import { AwsSolutionsChecks } from "cdk-nag";

import { MyStack } from "./stacks/my-stack";
import { configSchema, getEnvConfig } from "./utils/config";

const app = new App();
const config = getEnvConfig(app, configSchema);

const { service, environment } = config;

async function main() {
  const cloudflareCidrRanges = await getCloudflareCidrRanges(app);

  new MyStack(app, `${service}-mystack-${environment}`, {
    service,
    environment,
  });

  Aspects.of(app).add(new FluxSolutionsChecks());
  Aspects.of(app).add(new AwsSolutionsChecks());
  Aspects.of(app).add(new ApplyFluxStandardTags(config));
}

main();
