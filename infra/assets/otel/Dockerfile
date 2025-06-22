# https://gallery.ecr.aws/aws-observability/aws-otel-collector
FROM public.ecr.aws/aws-observability/aws-otel-collector:v0.40.0 AS aws-otel
FROM otel/opentelemetry-collector-contrib:0.119.0 AS otel-contrib

COPY --from=aws-otel /healthcheck /healthcheck

COPY config/otel-collector-config.yaml /etc/otelcol-contrib/config.yaml

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "/healthcheck" ]