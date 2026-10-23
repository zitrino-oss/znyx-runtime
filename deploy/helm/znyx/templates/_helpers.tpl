{{/*
Chart name (the app.kubernetes.io/name label), overridable with nameOverride.
*/}}
{{- define "znyx.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Release fullname: the base of every resource name (<name>, <name>-config, <name>-secrets).
*/}}
{{- define "znyx.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Selector labels: stable for the life of the release (a Deployment selector is immutable).
*/}}
{{- define "znyx.selectorLabels" -}}
app.kubernetes.io/name: {{ include "znyx.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Common labels on every resource.
*/}}
{{- define "znyx.labels" -}}
{{ include "znyx.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/component: gateway
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Names of the ConfigMap and the Secret in use.
*/}}
{{- define "znyx.configMapName" -}}
{{- printf "%s-config" (include "znyx.fullname" .) }}
{{- end }}

{{- define "znyx.secretName" -}}
{{- if .Values.existingSecret }}
{{- .Values.existingSecret }}
{{- else }}
{{- printf "%s-secrets" (include "znyx.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Container image: [registry/]repository:tag, with an "-ml" suffix on the tag when ml.enabled.
*/}}
{{- define "znyx.image" -}}
{{- $azure := dig "azure" "images" "runtime" dict .Values.global -}}
{{- if $azure -}}
{{- $registry := required "global.azure.images.runtime.registry is required" $azure.registry -}}
{{- $image := required "global.azure.images.runtime.image is required" $azure.image -}}
{{- if $azure.digest -}}
{{- printf "%s/%s@%s" (trimSuffix "/" $registry) (trimPrefix "/" $image) $azure.digest -}}
{{- else -}}
{{- printf "%s/%s:%s" (trimSuffix "/" $registry) (trimPrefix "/" $image) (required "Azure image needs a tag or digest" $azure.tag) -}}
{{- end -}}
{{- else -}}
{{- $tag := toString (default .Chart.AppVersion .Values.image.tag) -}}
{{- if and .Values.ml.enabled (not (hasSuffix "-ml" $tag)) -}}
{{- $tag = printf "%s-ml" $tag -}}
{{- end -}}
{{- $repo := .Values.image.repository -}}
{{- if .Values.image.registry -}}
{{- $repo = printf "%s/%s" (trimSuffix "/" .Values.image.registry) (trimPrefix "/" $repo) -}}
{{- end -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" $repo .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end }}
{{- end }}

{{/*
Values read out of the free-form config map (znyx.yaml): the container port, host and mode.
*/}}
{{- define "znyx.port" -}}
{{- (dig "server" "port" 8080 .Values.config) | int -}}
{{- end }}

{{- define "znyx.host" -}}
{{- dig "server" "host" "0.0.0.0" .Values.config -}}
{{- end }}

{{- define "znyx.mode" -}}
{{- dig "mode" "local" .Values.config -}}
{{- end }}

{{/*
Redis shared state is configured: the chart-created Secret holds a URL, or an existing
Secret is declared to hold one (state.redis.enabled). Renders "true" or nothing.
*/}}
{{- define "znyx.redisConfigured" -}}
{{- if .Values.existingSecret -}}
{{- if .Values.state.redis.enabled -}}true{{- end -}}
{{- else if .Values.secrets.redisUrl -}}
true
{{- end -}}
{{- end }}

{{/*
The replica count that matters for availability: the HPA floor when autoscaling.
*/}}
{{- define "znyx.minReplicas" -}}
{{- if .Values.autoscaling.enabled -}}
{{- .Values.autoscaling.minReplicas | int -}}
{{- else -}}
{{- .Values.replicaCount | int -}}
{{- end -}}
{{- end }}

{{/*
NetworkPolicy is rendered when asked for, and always when air-gapped.
*/}}
{{- define "znyx.networkPolicyEnabled" -}}
{{- if or .Values.networkPolicy.enabled .Values.airgapped -}}true{{- end -}}
{{- end }}

{{/*
Resource requests and limits: the user's, else the defaults for the mode.
*/}}
{{- define "znyx.resources" -}}
{{- if .Values.resources -}}
{{- toYaml .Values.resources -}}
{{- else if .Values.ml.enabled -}}
requests:
  cpu: "2"
  memory: 2Gi
limits:
  cpu: "4"
  memory: 4Gi
{{- else -}}
requests:
  cpu: "1"
  memory: 512Mi
limits:
  cpu: "2"
  memory: 1Gi
{{- end -}}
{{- end }}

{{/*
Fail early on combinations that cannot work, with a message that says what to change.
*/}}
{{- define "znyx.validate" -}}
{{- if eq (include "znyx.mode" .) "managed" -}}
{{- if not (dig "control_plane" "url" "" .Values.config) -}}
{{- fail "config.mode is managed: set config.control_plane.url to your console URL" -}}
{{- end -}}
{{- if and (not .Values.existingSecret) (not .Values.secrets.runtimeToken) -}}
{{- fail "config.mode is managed: set secrets.runtimeToken (or existingSecret with a ZNYX_RUNTIME_TOKEN key)" -}}
{{- end -}}
{{- end -}}
{{- if and .Values.airgapped (dig "telemetry" "heartbeat" false .Values.config) -}}
{{- fail "airgapped contradicts config.telemetry.heartbeat: an air-gapped install makes no outbound calls" -}}
{{- end -}}
{{- if and .Values.ml.enabled (dig "ml" "url" "" .Values.config) -}}
{{- fail "ml.enabled starts the model server in the pod; config.ml.url points at one elsewhere. Use one or the other" -}}
{{- end -}}
{{- end }}
