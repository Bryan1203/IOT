import boto3
import os

# Directory to save the certificate files
output_dir = "certificates"
region_name = 'us-east-2'
# Path formatters for certificate and private key
certificate_formatter = "./certificates/device_{}/device_{}.certificate.pem"
key_formatter = "./certificates/device_{}/device_{}.private.pem"

# Create an IoT client
iot_client = boto3.client("iot", region_name=region_name)

# Get the list of things
response = iot_client.list_things()
things = response["things"]

# Loop through each thing
for thing in things:
    thing_name = thing["thingName"]
    
    try:
        # Get the list of certificates associated with the thing
        response = iot_client.list_thing_principals(thingName=thing_name)
        principals = response["principals"]
        
        # Extract the certificate ID from the principals
        certificate_id = None
        for principal in principals:
            if principal.startswith("arn:aws:iot"):
                certificate_id = principal.split("/")[-1]
                break
        
        if certificate_id:
            # Get the certificate details
            cert_response = iot_client.describe_certificate(certificateId=certificate_id)
            cert_pem = cert_response["certificateDescription"]["certificatePem"]

            # Get the private key
            key_response = iot_client.get_registration_code()
            private_key_pem = key_response["registrationCode"]

            # Create the output directory for the thing if it doesn't exist
            thing_dir = os.path.join(output_dir, f"device_{thing_name}")
            os.makedirs(thing_dir, exist_ok=True)

            # Save the certificate file
            cert_file = certificate_formatter.format(thing_name, thing_name)
            with open(cert_file, "w") as file:
                file.write(cert_pem)

            # Save the private key file
            key_file = key_formatter.format(thing_name, thing_name)
            with open(key_file, "w") as file:
                file.write(private_key_pem)

            print(f"Certificate and private key files for thing '{thing_name}' saved to: {thing_dir}")
        else:
            print(f"No certificate found for thing: {thing_name}")
    
    except Exception as e:
        print(f"Error fetching certificate and private key for thing '{thing_name}': {str(e)}")

print("Certificate and private key fetching completed.")