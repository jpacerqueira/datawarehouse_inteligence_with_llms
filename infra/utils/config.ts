import { z } from "zod";
import { getContext } from "@sage/flux-cdk-lib/config";
import { App } from "aws-cdk-lib";

export const configSchema = z.object({
  awsAccountId: z.string(),
  awsRegion: z.string(),
  environment: z.string(),
  service: z.string(),
});

type ConfigEnv = z.infer<typeof configSchema>;

export function getEnvConfig(app: App, schema: z.AnyZodObject): ConfigEnv {
  const environment = app.node.tryGetContext("config-env");
  if (!environment) {
    throw new Error(
      "Context variable missing on CDK command. Pass in as -c config-env=<ENV>",
    );
  }
  return getContext<ConfigEnv>(app, schema, environment);
}
