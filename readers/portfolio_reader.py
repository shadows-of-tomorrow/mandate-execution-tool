import os
import pandas as pd

from instruments.portfolio import Portfolio
from instruments.factory import InstrumentFactory


class PortfolioReader:

    def __init__(self):
        self.portfolio_csv = "portfolio.csv"
        self.instrument_factory = InstrumentFactory()

    def read_portfolio(self, portfolio_path: str) -> Portfolio:
        # Todo: Another very ugly reader.
        instruments = []
        portfolio_path = os.path.join(portfolio_path, self.portfolio_csv)
        portfolio_df = pd.read_csv(portfolio_path)
        for k in range(len(portfolio_df)):
            identifier = portfolio_df.iloc[k, 0]
            instrument_type = portfolio_df.iloc[k, 1]
            discount_curve_quote = portfolio_df.iloc[k, 2]
            discount_curve_base = portfolio_df.iloc[k, 3]
            forecast_curve_quote = portfolio_df.iloc[k, 4]
            forecast_curve_base = portfolio_df.iloc[k, 5]
            notional = portfolio_df.iloc[k, 6]
            start_date = pd.to_datetime(portfolio_df.iloc[k, 7])
            maturity_date = pd.to_datetime(portfolio_df.iloc[k, 8])
            quote_currency = portfolio_df.iloc[k, 9]
            base_currency = portfolio_df.iloc[k, 10]
            ticker_symbol = portfolio_df.iloc[k, 11]
            kwargs = {}
            if not pd.isnull(discount_curve_quote):
                if not pd.isnull(discount_curve_base):
                    kwargs['discount_curve_quote_id'] = discount_curve_quote
                    kwargs['discount_curve_base_id'] = discount_curve_base
                else:
                    kwargs['discount_curve_id'] = discount_curve_quote
            if not pd.isnull(forecast_curve_quote):
                if not pd.isnull(forecast_curve_base):
                    kwargs['forecast_curve_quote_id'] = forecast_curve_quote
                    kwargs['forecast_curve_base_id'] = forecast_curve_base
                else:
                    kwargs['forecast_curve_id'] = forecast_curve_quote
            if not pd.isnull(notional):
                kwargs['notional'] = notional
            if not pd.isnull(start_date):
                kwargs['start_date'] = start_date
            if not pd.isnull(maturity_date):
                kwargs['maturity_date'] = maturity_date
            kwargs['quote_currency'] = quote_currency
            if not pd.isnull(base_currency):
                kwargs['base_currency'] = base_currency
            if not pd.isnull(ticker_symbol):
                kwargs['ticker_symbol'] = ticker_symbol
            instrument = self.instrument_factory.create_instrument(instrument_type, **kwargs)
            instruments.append(instrument)
        return Portfolio(instruments)
