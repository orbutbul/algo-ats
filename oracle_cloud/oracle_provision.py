"""
oracle_cloud/oracle_provision.py — retries creating an Oracle Always Free compute
instance until capacity is available, cycling through every availability
domain in the region each pass. "Out of host capacity" is the one error
this script treats as retryable (alongside 429 rate-limiting, backed off
separately -- see _is_rate_limit_error); anything else (bad OCID, quota
exceeded, auth failure) aborts immediately rather than looping forever on a
mistake.

Defaults to VM.Standard.E2.1.Micro (AMD x86_64, 1 OCPU/1GB, up to 2 free) --
Oracle's A1.Flex shape (ARM, 2 OCPU/12GB) is the higher-spec free option but
has been observed out of capacity for 5+ hours straight across every AD in
us-ashburn-1; Micro is essentially always available. Pass --shape
VM.Standard.A1.Flex to go back to trying for A1 instead.

One-time setup before running this:
    pip install oci
    oci setup config        # interactive: generates an API key pair, writes
                             # ~/.oci/config -- upload the printed public key
                             # under Identity -> My Profile -> API Keys in the
                             # OCI console (the wizard tells you this too)

You'll also need, from the OCI console:
    --compartment-id   Identity -> Compartments (root compartment OCID is
                        fine, shown in Tenancy details, if you didn't make one)
    --subnet-id        Networking -> Virtual Cloud Networks -> ats-trading-vcn
                        -> Subnets -> public subnet-ats-trading-vcn
    --ssh-key-file      path to the PUBLIC key (.pub) matching the private key
                        you'll SSH in with

Usage:
    python oracle_cloud/oracle_provision.py \\
        --compartment-id ocid1.tenancy.oc1..xxx \\
        --subnet-id ocid1.subnet.oc1..xxx \\
        --ssh-key-file ~/.ssh/ats_trading.pub

Leave it running (foreground or in a background terminal) -- it polls every
--interval seconds (default 120s) until it succeeds, hits --max-attempts, or
you Ctrl+C it. On success it sends a push notification via the same
validation.send_alert/ntfy.sh mechanism the pipeline already uses for
alerts, so you don't have to watch the terminal.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import oci

sys.path.insert(0, str(Path(__file__).parent.parent))
from validation import send_alert  # noqa: E402

UBUNTU_OS = 'Canonical Ubuntu'
UBUNTU_VERSION_PREFIX = '24.04'


def _latest_ubuntu_image_id(compute_client: 'oci.core.ComputeClient', compartment_id: str, shape: str) -> str:
    images = compute_client.list_images(
        compartment_id=compartment_id,
        operating_system=UBUNTU_OS,
        shape=shape,
        sort_by='TIMECREATED',
        sort_order='DESC',
    ).data
    for image in images:
        if image.operating_system_version.startswith(UBUNTU_VERSION_PREFIX):
            return image.id
    raise RuntimeError(f'No {UBUNTU_OS} {UBUNTU_VERSION_PREFIX} image found for {shape} in this compartment')


def _is_capacity_error(e: 'oci.exceptions.ServiceError') -> bool:
    return 'out of host capacity' in (e.message or '').lower() or e.status == 500 and 'capacity' in (e.message or '').lower()


def _is_rate_limit_error(e: 'oci.exceptions.ServiceError') -> bool:
    # Oracle throttles repeated LaunchInstance capacity-probing after enough
    # attempts (observed after ~3.7h of polling every 60s across 3 ADs) --
    # expected background noise for this workload, not a config mistake, so
    # it's retried too, just with a much longer backoff than a plain
    # "out of capacity" response to actually get under the rate limit.
    return e.status == 429


def provision(
    compartment_id: str,
    subnet_id: str,
    ssh_key_file: str,
    display_name: str,
    shape: str,
    ocpus: int,
    memory_gbs: int,
    interval: int,
    max_attempts: int | None,
    rate_limit_backoff: int,
) -> None:
    config = oci.config.from_file()
    identity_client = oci.identity.IdentityClient(config)
    compute_client = oci.core.ComputeClient(config)

    ads = [
        ad.name for ad in identity_client.list_availability_domains(compartment_id=compartment_id).data
    ]
    print(f'Availability domains in region: {ads}')

    image_id = _latest_ubuntu_image_id(compute_client, compartment_id, shape)
    print(f'Using image: {image_id}')

    ssh_public_key = Path(ssh_key_file).expanduser().read_text().strip()

    # Fixed shapes like VM.Standard.E2.1.Micro have no configurable
    # ocpus/memory -- shape_config is only valid for .Flex shapes.
    shape_config = (
        oci.core.models.LaunchInstanceShapeConfigDetails(ocpus=ocpus, memory_in_gbs=memory_gbs)
        if shape.endswith('.Flex') else None
    )

    launch_details = oci.core.models.LaunchInstanceDetails(
        compartment_id=compartment_id,
        display_name=display_name,
        shape=shape,
        shape_config=shape_config,
        source_details=oci.core.models.InstanceSourceViaImageDetails(image_id=image_id),
        create_vnic_details=oci.core.models.CreateVnicDetails(subnet_id=subnet_id, assign_public_ip=True),
        metadata={'ssh_authorized_keys': ssh_public_key},
    )

    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        for ad in ads:
            attempt += 1
            print(f'[attempt {attempt}] trying {ad}...')
            launch_details.availability_domain = ad
            try:
                response = compute_client.launch_instance(launch_details)
                instance = response.data
                print(f'Instance created: {instance.id} (state: {instance.lifecycle_state})')
                send_alert(
                    '[ATS] Oracle Cloud instance created',
                    f'Instance {display_name} launched in {ad} after {attempt} attempts.\n'
                    f'ID: {instance.id}\n'
                    f'Check the console for its public IP once RUNNING.',
                )
                return
            except oci.exceptions.ServiceError as e:
                if _is_capacity_error(e):
                    print(f'  {ad}: out of capacity, trying next domain/interval')
                    continue
                if _is_rate_limit_error(e):
                    print(f'  Rate limited (429) -- backing off {rate_limit_backoff}s before resuming')
                    time.sleep(rate_limit_backoff)
                    continue
                print(f'  Non-capacity error, aborting: {e.status} {e.code} {e.message}')
                raise
        print(f'All availability domains out of capacity. Sleeping {interval}s...')
        time.sleep(interval)

    print(f'Gave up after {max_attempts} attempts.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--compartment-id', required=True)
    p.add_argument('--subnet-id', required=True)
    p.add_argument('--ssh-key-file', required=True, help='path to the PUBLIC key (.pub)')
    p.add_argument('--display-name', default='ats_trading')
    p.add_argument('--shape', default='VM.Standard.E2.1.Micro', help='e.g. VM.Standard.E2.1.Micro (default, always-available) or VM.Standard.A1.Flex')
    p.add_argument('--ocpus', type=int, default=2, help='ignored for fixed (non-.Flex) shapes')
    p.add_argument('--memory-gbs', type=int, default=12, help='ignored for fixed (non-.Flex) shapes')
    p.add_argument('--interval', type=int, default=300, help='seconds between full AD sweeps (raised from an earlier 120s default -- that cadence got the account 429-rate-limited by Oracle after ~3.7h)')
    p.add_argument('--max-attempts', type=int, default=None, help='give up after this many tries (default: retry forever)')
    p.add_argument('--rate-limit-backoff', type=int, default=900, help='seconds to back off after a 429 TooManyRequests before resuming')
    args = p.parse_args()

    provision(
        args.compartment_id, args.subnet_id, args.ssh_key_file, args.display_name, args.shape,
        args.ocpus, args.memory_gbs, args.interval, args.max_attempts, args.rate_limit_backoff,
    )


if __name__ == '__main__':
    main()
