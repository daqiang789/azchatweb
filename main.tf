provider "oci" {}

resource "oci_core_instance" "generated_oci_core_instance" {
	agent_config {
		is_management_disabled = "false"
		is_monitoring_disabled = "false"
		plugins_config {
			desired_state = "DISABLED"
			name = "Vulnerability Scanning"
		}
		plugins_config {
			desired_state = "ENABLED"
			name = "Compute Instance Monitoring"
		}
		plugins_config {
			desired_state = "DISABLED"
			name = "Bastion"
		}
	}
	availability_config {
		recovery_action = "RESTORE_INSTANCE"
	}
	availability_domain = "YouW:AP-TOKYO-1-AD-1"
	compartment_id = "ocid1.tenancy.oc1..aaaaaaaatetksr7qw5o44znxbx33obw4m3yxzudfxhaaeqzgewtzjppgcbqq"
	create_vnic_details {
		assign_private_dns_record = "true"
		assign_public_ip = "true"
		subnet_id = "${oci_core_subnet.generated_oci_core_subnet.id}"
	}
	display_name = "instance-20211220-1513"
	instance_options {
		are_legacy_imds_endpoints_disabled = "false"
	}
	is_pv_encryption_in_transit_enabled = "true"
	metadata = {
		"ssh_authorized_keys" = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDGm5dy8u0TYXF2P3mD4rSZZ09VWDUWARgHaUVqWpMl+BOIeUF2+s32X0amv84t2x1PocFrpd8wbwG+N6SW6NW1GWn0cU/sjhHGE0ViVfE23Xvbm7lzZWLk/ZZQW84wy8vpcXIqsieiwia85jqwCoe2bVk1zdwll2V4e5tHBGFLbr/SssmOq7BOOEOIoSYHFqtH0+B3joQGWtZFci7XNOfYZN7qQdHMZATcxdhyMHg/LAaNmsY1j2zsOkNyZIpb4QTDL41vLkDTJV5m/oTDlu/TTfdoy8a/gArFdvEZEKr6m+1fF/D3Sa6CDoidr4Krc2B1jZBo7oWEXU9J2kvL+2gj ssh-key-2021-12-20"
	}
	shape = "VM.Standard.A1.Flex"
	shape_config {
		memory_in_gbs = "12"
		ocpus = "2"
	}
	source_details {
		source_id = "ocid1.image.oc1.ap-tokyo-1.aaaaaaaac2t4hu43w4fxidje6tou6wu5f5pujjvdyz2prad35hqwynh75akq"
		source_type = "image"
	}
}

resource "oci_core_vcn" "generated_oci_core_vcn" {
	cidr_block = "10.0.0.0/16"
	compartment_id = "ocid1.tenancy.oc1..aaaaaaaatetksr7qw5o44znxbx33obw4m3yxzudfxhaaeqzgewtzjppgcbqq"
	display_name = "vcn-20211220-1513"
	dns_label = "vcn12201520"
}

resource "oci_core_subnet" "generated_oci_core_subnet" {
	cidr_block = "10.0.0.0/24"
	compartment_id = "ocid1.tenancy.oc1..aaaaaaaatetksr7qw5o44znxbx33obw4m3yxzudfxhaaeqzgewtzjppgcbqq"
	display_name = "subnet-20211220-1513"
	dns_label = "subnet12201520"
	route_table_id = "${oci_core_vcn.generated_oci_core_vcn.default_route_table_id}"
	vcn_id = "${oci_core_vcn.generated_oci_core_vcn.id}"
}

resource "oci_core_internet_gateway" "generated_oci_core_internet_gateway" {
	compartment_id = "ocid1.tenancy.oc1..aaaaaaaatetksr7qw5o44znxbx33obw4m3yxzudfxhaaeqzgewtzjppgcbqq"
	display_name = "Internet Gateway vcn-20211220-1513"
	enabled = "true"
	vcn_id = "${oci_core_vcn.generated_oci_core_vcn.id}"
}

resource "oci_core_default_route_table" "generated_oci_core_default_route_table" {
	route_rules {
		destination = "0.0.0.0/0"
		destination_type = "CIDR_BLOCK"
		network_entity_id = "${oci_core_internet_gateway.generated_oci_core_internet_gateway.id}"
	}
	manage_default_resource_id = "${oci_core_vcn.generated_oci_core_vcn.default_route_table_id}"
}
