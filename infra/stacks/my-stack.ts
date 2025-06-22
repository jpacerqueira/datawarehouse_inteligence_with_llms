import { Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import { MyConstruct } from "../constructs/my-construct";

export interface MyStackProps extends StackProps {
  service: string;
  environment: string;
}

export class MyStack extends Stack {
  constructor(scope: Construct, id: string, props: MyStackProps) {
    super(scope, id, props);

    const { service, environment } = props;

    new MyConstruct(this, "MyConstruct", {
      service,
      environment,
      myConstructProp: "myConstructProp",
    });
  }
}
