from economy.observables.base import Observable


class InterestRate(Observable):
    """
    An interest rate is an amount charged, expressed as a percentage of principal,
    by a lender to a borrower for the use of assets. Its observable value is typically
    noted on an annual basis, known as the annual percentage rate (APR).
    """
    def __init__(self, identifier: str, currency: str, value: float) -> None:
        self.currency = currency
        super().__init__(identifier=identifier, value=value)
