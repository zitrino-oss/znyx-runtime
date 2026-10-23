# Marketplace seller review package

Azure is the first target and the builder's default. It prepares native Kubernetes Application
inputs: a Marketplace-compatible Helm chart, CPA manifest, ARM extension template and portal UI.
It does not create a product, upload an image, run an authenticated CNAB build, integrate paid
marketplace metering or publish a listing. AWS/GCP retain portable seller review archives.

Build and scan the release image, sign it through the release pipeline, push it to the actual
marketplace-assigned repository, and obtain its registry manifest digest. Then run:

```sh
python deploy/marketplace/build_delivery.py \
  --cloud azure \
  --image "$MARKETPLACE_IMAGE_AT_DIGEST" \
  --azure-application-name "$AZURE_APPLICATION_NAME" \
  --azure-publisher "$AZURE_PUBLISHER_DISPLAY_NAME" \
  --version 1.0.0 \
  --output /tmp/znyx-seller-review
```

`MARKETPLACE_IMAGE_AT_DIGEST` must be the real assigned repository followed by
`@sha256:<64 lowercase hex characters>`. Tags are refused; seller/product IDs are never fabricated.
For Azure, supply the actual `*.azurecr.io` image reference, chosen reverse-domain application
name and publisher display name. Versions have three numeric components. The image must contain
the Linux AMD64 media build (`docker build --platform linux/amd64 --target znyx-media ...`).
The existing ARM64 development image is not a Marketplace release image.

The output is `azure/`, `delivery.json`, `marketplace-values.json` and `znyx-delivery.tar.gz`.
`azure/manifest.yaml` points to the unpacked native chart, ARM template and portal definition.
The manifest records input image identity, generated/source file SHA-256s and the
explicit `requires-seller-submission` status. Tar/gzip metadata is deterministic. Checksums
are integrity evidence, not a release signature; sign the archive in the release pipeline.

The marketplace values use the pinned runtime image for both the service and Helm health
test, so deployment needs no external curl helper image. Overlay customer settings for the
control-plane URL, customer Secret, private Redis, ingress/egress and optional media model PVC.
Use the actual media image digest with `operated/media-values.yaml` for OCR/speech workloads.

The Azure chart resolves both images through `global.azure.images.runtime`, including when
Microsoft rehosts the image under a new registry/path/tag. The generated chart defaults to
two replicas, private TLS Redis, an existing shared data PVC and a read-only model PVC.
The portal requires the real control-plane URL, existing customer Secret/PVC names and Redis
private endpoint IP/port. It never collects credentials. Extension upgrades are pinned and
automatic minor upgrades are disabled. The reserved `DONOTMODIFY` plan/type variables follow
Microsoft's packaging contract and are populated by its publishing flow, not fabricated here.

## Azure seller validation and CNAB build

Use a supported Linux/Windows AMD64 packaging runner with Docker and the seller's real ACR and
Partner Center tenant. Follow [Microsoft's technical asset workflow](https://learn.microsoft.com/en-us/partner-center/marketplace-offers/azure-container-technical-assets-kubernetes)
to authorize registry ingestion and sign in. Use an approved digest of the official
`mcr.microsoft.com/container-package-app` tool. Mount the generated `azure/` directory as its
working content and the Docker socket as required by that tool. Inside its shell:

```sh
az login
az acr login --name "$REGISTRY_NAME"
cd /data
cpa verify
```

After seller review, `cpa buildbundle` verifies, builds **and uploads** the CNAB to ACR. It is
an external write and is not run by this builder or the local checks. Submit the resulting
real CNAB reference in Partner Center, run preview deployment/certification, and record the
product/plan/version/listing evidence. The local archive is a set of native CNAB inputs,
not an already built or certified CNAB. Use a new empty delivery directory for each Azure build.

Before submission, validate the customer installation and revocation runbook, scan all release
architectures, obtain legal/support/privacy/listing material and run the provider's acceptance
checks. Upload into assigned repositories and record the resulting product/version/listing IDs.
No such IDs or live publication evidence exist in this local implementation.

Cloud-specific requirements remain part of seller acceptance:

- AWS: standard container/Helm BYOL delivery; EKS add-on delivery does not support BYOL.
  All required images must be marketplace-hosted, and commercial BYOL editions have additional
  paid-edition requirements. Review the [container requirements](https://docs.aws.amazon.com/marketplace/latest/userguide/container-product-policies.html)
  and [pricing models](https://docs.aws.amazon.com/marketplace/latest/userguide/pricing-container-products.html)
  against the actual business offer before submitting.
- Azure: native inputs are generated here; authenticated CPA verification/build/upload and
  Partner Center publication remain seller acceptance. See [container offer requirements](https://learn.microsoft.com/en-us/partner-center/marketplace-offers/marketplace-containers)
  and the [Azure operated runbook](../operated/azure/README.md).
- GCP: produce the provider's native Kubernetes deployment/manifest assets with the assigned
  service identity and validate the licensing flow. See [Kubernetes apps](https://docs.cloud.google.com/marketplace/docs/partners/kubernetes)
  and [BYOL requirements](https://docs.cloud.google.com/marketplace/docs/partners/kubernetes/select-pricing).

The review archive selects BYOL as an initial packaging assumption. Public pricing, license
terms, paid marketplace billing and additional clouds require an explicit product decision.
