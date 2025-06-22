import { Construct } from "constructs";
import * as sqs from "aws-cdk-lib/aws-sqs";
import { Duration } from "aws-cdk-lib";

export interface MyConstructProps {
  service: string;
  environment: string;
  myConstructProp: string;
}

export class MyConstruct extends Construct {
  constructor(scope: Construct, id: string, props: MyConstructProps) {
    super(scope, id);

    const appQueueDlq = new sqs.Queue(this, "AppQueueDLQ", {
      enforceSSL: true,
      retentionPeriod: Duration.days(14),
    });

    new sqs.Queue(this, "AppQueue", {
      enforceSSL: true,
      deadLetterQueue: {
        maxReceiveCount: 3,
        queue: appQueueDlq,
      },
    });
  }
}
