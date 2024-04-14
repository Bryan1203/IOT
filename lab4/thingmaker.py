import boto3
import json
import random
import string

# Parameters
thing_name = ''.join(random.choices(string.ascii_letters + string.digits, k=15))
default_policy_name = '437policy'
region_name = 'us-east-2'
thing_group_name = '437group'
thing_group_arn = 'arn:aws:iot:us-east-2:891377320821:thinggroup/437group'

# Initialize Boto3 IoT client
iot_client = boto3.client('iot', region_name=region_name)

def create_thing(thing_name):
    """Create an IoT thing."""
    response = iot_client.create_thing(thingName=thing_name)
    thing_arn = response['thingArn']
    thing_id = response['thingId']
    return thing_arn, thing_id

def create_certificate():
    """Create keys and certificate."""
    response = iot_client.create_keys_and_certificate(setAsActive=True)
    certificate_arn = response['certificateArn']
    public_key = response['keyPair']['PublicKey']
    private_key = response['keyPair']['PrivateKey']
    certificate_pem = response['certificatePem']

    # Save keys and certificate to files
    with open('public.key', 'w') as outfile:
        outfile.write(public_key)
    with open('private.key', 'w') as outfile:
        outfile.write(private_key)
    with open('cert.pem', 'w') as outfile:
        outfile.write(certificate_pem)
    
    return certificate_arn

def attach_policy_and_principal(policy_name, thing_name, certificate_arn):
    """Attach a policy to the certificate and attach the thing to the certificate (principal)."""
    iot_client.attach_policy(policyName=policy_name, target=certificate_arn)
    iot_client.attach_thing_principal(thingName=thing_name, principal=certificate_arn)

def add_thing_to_thing_group(thing_group_name, thing_name):
    """Add the thing to a thing group."""
    iot_client.add_thing_to_thing_group(
        thingGroupName=thing_group_name,
        thingName=thing_name
    )

# Workflow
thing_arn, thing_id = create_thing(thing_name)
certificate_arn = create_certificate()
attach_policy_and_principal(default_policy_name, thing_name, certificate_arn)
add_thing_to_thing_group(thing_group_name, thing_name)

print(f"Thing {thing_name} created and added to group {thing_group_name}.")
