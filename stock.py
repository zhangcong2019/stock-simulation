import random
import math

import matplotlib.pyplot as plt
import numpy as np

random.seed()
days_per_year = 365
total_days = days_per_year * 5
days_init = np.min([1000, total_days])
growth_gdp = 0.06

std_daily_return = 0.01
elastic_price = 0.002
elastic_growth = 0.002
elastic_emotion = 0.002
growth_change_rate = 0.002
emotion_change_rate = 0.02
emotion_momentum = 0.95

proportion_equity = 0.5
num_company = 1000
trade_period = 30
commission_rate = 0.0001
return_debt = 0.05

emotion = 1.0

constant_growth = True
flag_plot = False


class Company:
    def __init__(self):
        self.price_inner = np.zeros(shape=total_days, dtype=np.float64)
        self.price_inner[0] = 100.0
        self.price = np.zeros(shape=total_days, dtype=np.float64)
        self.price[0] = self.price_inner[0]
        self.growth_rate = np.zeros(shape=total_days, dtype=np.float64)
        self.growth_rate[0] = 0
        self.ret = np.zeros(shape=total_days, dtype=np.float64)
        self.total_ret = 0.0
        self.emotion = np.zeros(shape=total_days, dtype=np.float64)
        self.emotion[0] = 1.0

        ran_emotion_change = 0.0
        for i in range(1, days_init):
            if constant_growth:
                self.price_inner[i] = self.price_inner[i - 1]
            else:
                ran_growth_change = (random.random() * 2 - 1) * growth_change_rate
                self.growth_rate[i] = self.growth_rate[i - 1] + ran_growth_change - elastic_growth * (
                    self.growth_rate[i - 1])
                self.price_inner[i] = self.price_inner[i - 1] * math.exp(math.log(1 + self.growth_rate[i]) \
                                                                         / days_per_year)

            ran_emotion_change = ran_emotion_change * emotion_momentum + (random.random() * 2 - 1) * emotion_change_rate * (1-emotion_momentum)
            self.emotion[i] = self.emotion[i - 1] * np.exp(ran_emotion_change) - elastic_emotion * (
                self.emotion[i - 1] - 1)
            # self.emotion[i] = self.emotion[i-1] * emotion_momentum + (1 - emotion_momentum) * self.emotion[i]

            self.ret[i] = random.gauss(0, std_daily_return)
            self.ret[i] = self.ret[i] + math.log(self.price_inner[i] / self.price_inner[i - 1]) \
                          + math.log(self.price_inner[i] / self.price[i - 1]) * elastic_price
            self.total_ret = self.total_ret + self.ret[i]
            self.price[i] = self.price[0] * np.exp(self.total_ret + np.log(self.emotion[i] / self.emotion[0]))

        self.price_inner[0] = self.price_inner[days_init - 1]
        self.price[0] = self.price[days_init - 1]
        self.growth_rate[0] = growth_gdp + self.growth_rate[days_init - 1]
        self.emotion[0] = self.emotion[days_init - 1]
        self.total_ret = 0.0

        for i in range(1, total_days):
            if constant_growth:
                self.price_inner[i] = self.price_inner[i - 1] * math.exp(math.log(1 + growth_gdp) / days_per_year)
            else:
                ran_growth_change = (random.random() - 0.5) * 2 * growth_change_rate
                self.growth_rate[i] = self.growth_rate[i - 1] + ran_growth_change - elastic_growth * (
                        self.growth_rate[i - 1] - growth_gdp)
                self.price_inner[i] = self.price_inner[i - 1] * math.exp(math.log(1 + self.growth_rate[i]) \
                                                                         / days_per_year)
            ran_emotion_change = ran_emotion_change * emotion_momentum + (
                        random.random() * 2 - 1) * emotion_change_rate * (1 - emotion_momentum)
            self.emotion[i] = self.emotion[i - 1] * np.exp(ran_emotion_change) - elastic_emotion * (
                self.emotion[i - 1] - 1)
            # self.emotion[i] = self.emotion[i-1] * emotion_momentum + (1 - emotion_momentum) * self.emotion[i]

            self.ret[i] = random.gauss(0, std_daily_return)
            self.ret[i] = self.ret[i] + math.log(self.price_inner[i] / self.price_inner[i - 1]) \
                          + math.log(self.price_inner[i] / self.price[i - 1]) * elastic_price
            self.total_ret = self.total_ret + self.ret[i]
            self.price[i] = self.price[0] * np.exp(self.total_ret + np.log(self.emotion[i] / self.emotion[0]))

        if flag_plot:
            x = np.linspace(0, total_days, total_days)
            if not constant_growth:
                plt.plot(x, self.growth_rate * 1000, 'r')
            plt.plot(x, self.price_inner, 'g')
            plt.plot(x, self.price, 'b')
            plt.plot(x, self.emotion * 100, 'y')
            plt.show()

    def trade(self, period):
        if period < 1:
            period = 1
        capital = 1.0
        emotion_temp = self.emotion[0]
        proportion_equity = 0.5
        capital_equity = capital * proportion_equity
        capital_debt = capital * (1 - proportion_equity)
        for i in range(period, total_days, period):
            emotion_temp = self.emotion[i] * 0.2 + emotion_temp * 0.8
            if emotion_temp > 1.3:
                proportion_equity = 0.25
            else:
                if emotion_temp < 0.77:
                    proportion_equity = 0.75
                else:
                    proportion_equity = 0.55
            capital_equity = capital_equity * self.price[i] / self.price[i - period]
            capital_debt = capital_debt * math.exp(math.log(1 + return_debt) * period / days_per_year)
            capital = capital_equity + capital_debt
            commission = 2 * commission_rate * math.fabs(capital * proportion_equity - capital_equity)
            capital = capital - commission
            capital_equity = capital * proportion_equity
            capital_debt = capital * (1 - proportion_equity)
        return capital

    @property
    def final_price(self):
        return self.price[total_days - 1]

    @property
    def final_inner_price(self):
        return self.price_inner[total_days - 1]

    @property
    def init_price(self):
        return self.price[0]

    @property
    def init_inner_price(self):
        return self.price_inner[0]


def main():
    capital = np.zeros(shape=num_company, dtype=np.float64)
    init_price = np.zeros(shape=num_company, dtype=np.float64)
    final_price = np.zeros(shape=num_company, dtype=np.float64)
    init_inner_price = np.zeros(shape=num_company, dtype=np.float64)
    final_inner_price = np.zeros(shape=num_company, dtype=np.float64)
    for i in range(0, num_company):
        company = Company()
        capital[i] = company.trade(trade_period)
        init_price[i] = company.init_price
        final_price[i] = company.final_price
        init_inner_price[i] = company.init_inner_price
        final_inner_price[i] = company.final_inner_price

    print("final capital mean is " + str(capital.mean()))
    print("final capital deviation is " + str(capital.std()))

    print("final init_price mean is " + str(init_price.mean()))
    print("final final_price mean is " + str(final_price.mean()))
    print("final init_inner_price mean is " + str(init_inner_price.mean()))
    print("final final_inner_price mean is " + str(final_inner_price.mean()))

    # x = np.linspace(0, num_company, num_company)
    #
    # plt.plot(x, capital, 'r')
    # plt.show()


if __name__ == '__main__':
    main()
