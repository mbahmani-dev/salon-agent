"""
Payment layer. The rest of the agent only talks to the two methods of PaymentProvider:

    create(...)  -> PaymentRequest(provider_ref, pay_url)     # start a payment, get a link for the customer
    verify(...)  -> VerifyResult(paid, bank_ref, amount)      # ask the gateway "was it really paid?"

To go live later: implement RealGateway below for your gateway, set PAYMENT_PROVIDER=real
and PAYMENT_CALLBACK_URL to your public https address. Nothing else has to change.

How a redirect-style online payment normally works (check your gateway's own docs for exact fields):
  1. Your server asks the gateway to create a payment (amount, description, callback URL).
     The gateway returns a reference/token. You build the pay link from it.
  2. The customer opens the link, pays on the BANK/GATEWAY page.
  3. The gateway sends the customer's browser back to your callback URL with a reference and a status.
  4. Your server must NOT trust that redirect. It calls the gateway server-to-server (verify),
     checks it was paid AND that the amount is the one stored in your database, and only then
     confirms the booking.
  5. The callback can arrive twice or late: the code must be idempotent.
"""
import os
import secrets
from dataclasses import dataclass
from typing import Optional

CALLBACK_BASE_URL = os.environ.get("PAYMENT_CALLBACK_URL", "http://localhost:8000")


@dataclass
class PaymentRequest:
    provider_ref: str          # id the gateway gives us (we store it to find the payment later)
    pay_url: str               # link the customer opens to pay


@dataclass
class VerifyResult:
    paid: bool
    bank_ref: Optional[str] = None       # bank tracking code (show it to the customer / keep for refunds)
    amount_toman: Optional[int] = None   # amount the gateway says was paid (None if it does not tell)
    message: str = ""


class PaymentProvider:
    name = "base"

    def create(self, *, booking_id: int, amount_toman: int, description: str, callback_url: str) -> PaymentRequest:
        raise NotImplementedError

    def verify(self, *, provider_ref: str, expected_amount_toman: int, callback_params: dict) -> VerifyResult:
        raise NotImplementedError


class FakeGateway(PaymentProvider):
    """Development only. 'Pays' when the callback says status=OK. Never use with real money:
    anyone could forge that parameter."""
    name = "fake"

    def create(self, *, booking_id, amount_toman, description, callback_url) -> PaymentRequest:
        ref = "FAKE-" + secrets.token_hex(4).upper()
        return PaymentRequest(provider_ref=ref, pay_url=f"{callback_url.rstrip('/')}/fake-pay/{ref}")

    def verify(self, *, provider_ref, expected_amount_toman, callback_params) -> VerifyResult:
        if callback_params.get("status") == "OK":
            return VerifyResult(True, bank_ref="FAKE-" + secrets.token_hex(3).upper(),
                                amount_toman=expected_amount_toman)
        return VerifyResult(False, message="payment cancelled or failed")


class RealGateway(PaymentProvider):
    """TEMPLATE for the real gateway (Zarinpal, IDPay, Zibal, ... whichever you choose).
    Fill in the two methods using that gateway's documentation."""
    name = "real"

    def create(self, *, booking_id, amount_toman, description, callback_url) -> PaymentRequest:
        # 1) POST the amount, description and callback_url to the gateway's "request payment" endpoint
        #    (mind whether the gateway wants Toman or Rial!)
        # 2) read the reference/token from its answer
        # 3) return PaymentRequest(provider_ref=<that reference>, pay_url=<the gateway's pay link for it>)
        raise NotImplementedError("fill in with your gateway's 'request payment' call")

    def verify(self, *, provider_ref, expected_amount_toman, callback_params) -> VerifyResult:
        # 1) if the gateway reports the customer cancelled -> return VerifyResult(False, ...)
        # 2) call the gateway's "verify" endpoint with provider_ref and the expected amount
        # 3) return VerifyResult(True, bank_ref=<tracking code>, amount_toman=<amount it confirms>)
        raise NotImplementedError("fill in with your gateway's 'verify payment' call")


def get_provider() -> PaymentProvider:
    name = os.environ.get("PAYMENT_PROVIDER", "fake").lower()
    if name == "fake":
        return FakeGateway()
    if name == "real":
        return RealGateway()
    raise ValueError(f"unknown PAYMENT_PROVIDER '{name}'")
