"""Salted-hash commitment C = SHA256(node_id | draw | quota | salt) + compliance check w <= Qmax.

NOTE: this is a hash commitment, not a true zero-knowledge proof. Be ready to explain the
difference in the oral defense.
"""
import hashlib
import secrets

from pydantic import BaseModel, Field

from agent.skills.base import BaseAgentSkill


def compute_commitment(node_id: str, draw: float, quota: float, salt: str) -> str:
    payload = f"{node_id}|{draw:.4f}|{quota:.4f}|{salt}".encode()
    return "0x" + hashlib.sha256(payload).hexdigest()  # 66 chars = bytes32 / model max_length


def verify_commitment(node_id: str, draw: float, quota: float, salt: str, commitment: str) -> bool:
    return secrets.compare_digest(compute_commitment(node_id, draw, quota, salt), commitment)


class ZKInput(BaseModel):
    node_id: str
    water_draw: float = Field(ge=0)
    quota_limit: float = Field(gt=0)
    salt: str | None = None  # supplied only in tests / re-verification


class ZKOutput(BaseModel):
    compliant: bool
    commitment_hash: str | None
    salt: str | None            # PRIVATE: keep off-chain
    public_inputs: dict


class ZKWaterComplianceProverSkill(BaseAgentSkill):
    name = "zk_water_compliance_prover"
    description = "Proves water draw <= quota via a salted commitment without exposing plot data."
    input_model, output_model = ZKInput, ZKOutput

    def execute(self, data: ZKInput) -> ZKOutput:
        compliant = data.water_draw <= data.quota_limit
        public = {"quota_limit": data.quota_limit, "compliant": compliant}
        if not compliant:  # never anchor a breach
            return ZKOutput(compliant=False, commitment_hash=None, salt=None, public_inputs=public)
        salt = data.salt or secrets.token_hex(16)
        commitment = compute_commitment(data.node_id, data.water_draw, data.quota_limit, salt)
        return ZKOutput(compliant=True, commitment_hash=commitment, salt=salt, public_inputs=public)
