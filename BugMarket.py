
import asyncio
from concurrent.futures import ProcessPoolExecutor
import pandas
import random
import matplotlib
import os
import datetime
import pickle
import zlib 

import redis

RISK_LOWER = 40
RISK_UPPER = 50
PERSISTENT = True
stonks = None

class stock_engine():
    MARKET_VOLATILENESS = 15  # 0-100 0 being C4, 100 being a politcal discussion during thanksgiving 
    STOCK_SAVE_FP = '/home/pi/workspace/Botty/deps/stockMarket.txt'
    STOCK_BACKUP_FP = '/home/pi/workspace/Botty/deps/stockMarket_back.txt'
    STOCKS = {
        'MSC': 'Mischief Co.',
        'CCI':'ContoCorp Inc.',
        'TCM':'The Crepe Monopoly',
        'UWU':'United Whites Union',
        'KEK':'Konnected Enterprises of Kama',
        'BOT':'Botty Corp Inc LLT LLC (Tm) Patent Pending Copyright 2020'
        }
    running = False
    marketGrowthFactor = None  # between -1 and 1
    config = None
    stocks = {}
    investors = {}
    tick = 0

    trendMessages = [None, None]
    trendMessagesCount = 0
    
    def __init__(self):
        self.config = {'ticks': 2}  # Ticks = calculations per minute
        self.marketGrowthFactor = .1

        if PERSISTENT:
            if os.path.exists(self.STOCK_SAVE_FP):
                with open(self.STOCK_SAVE_FP, 'r') as f:
                    data = f.readline().strip().split('|')
                    self.tick = int(data[0])
                    self.marketGrowthFactor = float(data[1])
                    for line in f.readlines():
                        data = line.strip().split('|')
                        if data[0] == 'stock':
                            self.stocks[data[2]] = stock(data[1], data[2], float(data[3]), int(data[4]), data[5])
                        elif data[0] == 'investor':
                            self.investors[int(data[2])] = investor(data[1], int(data[2]), float(data[3]), data[4], data[5], data[6], data[7])
            else:
                for sts, st in self.STOCKS.items():
                    self.stocks[sts] = stock(st, sts, random.randint(150,250), random.randint(RISK_LOWER, RISK_UPPER))
        else:
            for sts, st in self.STOCKS.items():
                self.stocks[sts] = stock(st, sts, random.randint(150,250), random.randint(RISK_LOWER, RISK_UPPER))

    async def startMarket(self):
        if not self.running:
            self.running = True
            await self.updateMarket()

    async def stopMarket(self):
        self.running = False

    async def updateMarket(self):
        while self.running:
            
            if PERSISTENT: #  Save every tick
                await self.save()
                self.running = True
            await asyncio.sleep(60 / int(self.config['ticks']))
            if random.randint(0,100) < self.MARKET_VOLATILENESS:
                rfactor = 1 if (random.randint(0,100) > 50 * (1 + 16*(self.marketGrowthFactor * abs(self.marketGrowthFactor)))) else -1 
                self.marketGrowthFactor += rfactor * random.randint(1,3) * .01
                #print(self.marketGrowthFactor)
            self.tick += 1
            for st in self.stocks.values():
                st.updatePrice(self.tick, self.marketGrowthFactor)

            for inv in self.investors.values():
                for lim in inv.limits:
                    if lim['type'] == 'buy':
                        if self.stocks[lim['stock']].price <= lim['limit']:
                            inv.buy(lim['stock'], lim['amount'], lim['limit'])
                            inv.limits.remove(lim)
                    else:
                        if self.stocks[lim['stock']].price >= lim['limit']:
                            inv.buy(lim['stock'], lim['amount'], lim['limit'])
                            inv.limits.remove(lim)
                stocksPrice = 0
                for stk in inv.portfolio:
                    stocksPrice += self.stocks[stk].price * inv.portfolio[stk]
                inv.update(self.tick, stocksPrice)
            
            #print(self.trendMessages)
            for msg in self.trendMessages:
                if msg is not None:
                    rtn = await self.showTrends()
                    await msg.edit(content=rtn)
            print(self.tick)
        print('exit')
            

    async def showTrends(self):
        if self.tick < 100:
            return {'error': 'Not enough ticks have passed.'}
        msg = []
        msg.append('```ini\n')
        msg.append(f'Tick: {self.tick}  Last Updated: {datetime.datetime.now()}')
        msg.append(f'[Stock  Price  1-Tick  10-Tick  100-Tick]')
        for st in self.stocks.values():
            prices = list(st.priceHistory.values())[-100:]
            msg.append(f'[{st.short_name}]  ${st.price:4.2f} {self.sign(prices[0], prices[1])}${abs(prices[1] - prices[0]):.2f}  {self.sign(prices[0], prices[9])}${abs(prices[9] - prices[0]):.2f}   {self.sign(prices[0], prices[99])}${abs(prices[99] - prices[0]):.2f}')
        msg.append('\n```')

        return {'msg': '\n'.join(msg), 'timeUntilNextUpdate': 60 / int(self.config['ticks'])}

    def sign(self, p1, p2):
        return '-' if p2 - p1 < 0 else ' '

    def addTrendMsg(self, msg):
        self.trendMessages[self.trendMessagesCount % 2] = msg
        self.trendMessagesCount += 1

    async def showInvestors(self):
        msg = []
        msg.append('```ini\n')

        for inv in self.investors.values():
            stocks = 0
            for stk in inv.portfolio:
                stocks += self.stocks[stk].price * inv.portfolio[stk]
            msg.append(f'[{inv.name:^25}] Total ${inv.money + stocks:4.2f}\tCash ${inv.money:4.2f}\tStocks ${stocks:4.2f}')
        msg.append('\n```')

        return {'msg': '\n'.join(msg)}

    async def showMarket(self, stock='all', time=0):
        if time == 0:
            time = self.tick
        showList = []
        if stock.lower() == 'all':
            showList = self.STOCKS.keys()
        elif ':' in stock:
            stks = stock.split(':')
            for stk in stks:
                if stk.upper() in self.STOCKS.keys():
                    showList.append(stk.upper())
                else:
                    return {'error':f'{stk} is not real dummyhead'}
        else:
            if stock.upper() in self.STOCKS.keys():
                    showList.append(stock.upper())
            else:
                return {'error':f'{stock} is not real dummyhead'}
        marketGraph = '/home/pi/workspace/Botty/downloads/market.png'
        msg = []
        msg.append('```ini\n')
        data = {}

        for name in showList:
            msg.append(f'{self.stocks[name].name:60}  [${self.stocks[name].price:.2f}]')
            data[name] = list(self.stocks[name].priceHistory.values())
        df = pandas.DataFrame(data, index=list(range(0, self.tick + 1)))
        df = df.loc[df.index > self.tick + 1 - time]
        graph = df.plot.line()
        fig = graph.get_figure()
        fig.savefig(marketGraph)
        matplotlib.pyplot.close(fig)
        msg.append('\n```')

        return {'msg':'\n'.join(msg), 'graph':marketGraph}


    async def lossPorn(self, uid, time=0):
        if time == 0:
            time = self.tick

        if uid in self.investors:
            lossGraph = '/home/pi/workspace/Botty/downloads/lossporn.png'
            data = {}

            data[self.investors[uid].name] = self.investors[uid].history

            df = pandas.DataFrame(data, index=list(range(0, self.tick + 1)))
            df = df.loc[df.index > self.tick + 1 - time]
            graph = df.plot.line()
            fig = graph.get_figure()
            fig.savefig(lossGraph)
            matplotlib.pyplot.close(fig)

            return {'msg': '', 'graph':lossGraph}
        else:
            return {'error': 'Make an account ya goober'}

    async def showHoldings(self, uid):
        if uid in self.investors:
            msg = []
            msg.append('```ini\n')
            msg.append(f'Cash: [${self.investors[uid].money:.2f}]')
            for stk in self.investors[uid].portfolio:
                msg.append(f'[{self.investors[uid].portfolio[stk]}] shares of [{stk}] worth [${self.stocks[stk.upper()].price:.2f}] each for a total of [${self.stocks[stk.upper()].price * self.investors[uid].portfolio[stk]:.2f}]')
            msg.append('\n```')
            return {'msg': '\n'.join(msg)}
        else:
            return {'error': 'Make an account ya goober'}

    async def save(self):
        self.running = False 
        #await asyncio.sleep(60 / int(self.config['ticks']))  # Ensure that the last tick finishes before we save off data
        if os.path.exists(self.STOCK_SAVE_FP):
            os.rename(self.STOCK_SAVE_FP, self.STOCK_BACKUP_FP)

        with open(self.STOCK_SAVE_FP, 'w+') as f:
            f.write(f'{self.tick}|{self.marketGrowthFactor}\n')
            for stk in self.stocks.values():
                f.write(f'stock|{stk.name}|{stk.short_name}|{stk.price}|{stk.riskFactor}|{stk.priceHistory}\n')
            for inv in self.investors.values():
                f.write(f'investor|{inv.name}|{inv.uid}|{inv.money}|{inv.portfolio}|{inv.transactions}|{inv.history}|{inv.limits}\n')

    async def openAccount(self, name, uid):
        if int(uid) in self.investors.keys():
            return {'error': 'Account already Exists'}
        self.investors[int(uid)] = (investor(name, uid))
        return {'msg': 'Welcome Investor!'}

    async def buy(self, uid, stk, amount:int):
        if uid not in self.investors.keys():
            return {'error': 'You need to open an account dumdum'}
        if stk in self.stocks.keys():
            return self.investors[uid].buy(stk, amount, self.stocks[stk].price)
        else:
            return {'error': 'Try a real stock'}

    async def sell(self, uid, stk, amount):
        if uid not in self.investors.keys():
            return {'error': 'You need to open an account dumdum'}
        if stk in self.stocks.keys():
            return self.investors[uid].sell(stk, amount, self.stocks[stk].price)
        elif stk.lower() == 'all':
            for stock in self.investors[uid].portfolio.copy():
                msg = self.investors[uid].sell(stock, 'all', self.stocks[stock].price)
            return msg
        else:
            return {'error': 'Try a real stock'}

    async def limitBuy(self, uid, stk, amount:int, limit:float):
        if uid not in self.investors.keys():
            return {'error': 'You need to open an account dumdum'}
        if stk in self.stocks.keys():
            return self.investors[uid].limitBuy(stk, int(amount), float(limit))

    async def limitSell(self, uid, stk, amount, limit):
        if uid not in self.investors.keys():
            return {'error': 'You need to open an account dumdum'}
        if stk in self.stocks.keys():
            return self.investors[uid].limitSell(stk, int(amount), float(limit))

    async def showLimits(self, uid):
        if uid not in self.investors.keys():
            return {'error': 'You need to open an account dumdum'}

        msg = []
        msg.append('```ini\n')

        if len(self.investors[uid].limits) == 0:
            return {'msg': 'You have no limits'}

        for lim, i in zip(self.investors[uid].limits, range(0, len(self.investors[uid].limits))):
            msg.append(f"{i}: [{lim['type']} {lim['amount']} {lim['stock']} at {lim['limit']}]")
        
        msg.append('\n```')

        return {'msg': '\n'.join(msg)}

    async def cancelLimit(self, uid, index):
        if index + 1 <= len(self.investors[uid].limits):
            return self.investors[uid].limits.pop(index)
        else:
            return {'error': 'Not a valid index'}


class stock():
    riskFactor = None  # 1-100 1 being extremely safe, 100 being extrememly dangerous
    price = None
    priceHistory = None  # {tick: price}
    newsHistory = None
    name = None
    short_name = None

    def __init__(self, name, short_name, price, risk, priceHistory=None):
        self.name = name
        self.short_name = short_name
        self.price = price
        self.riskFactor = risk
        if priceHistory:
            exec(f'self.priceHistory = {priceHistory}')
        else:
            self.priceHistory = {0: self.price}
        self.newsHistory = []

    def updatePrice(self, tick, mgf):
        if random.randint(0, 100) < 10:
            self.riskFactor += random.choice([1, -1])
            if self.riskFactor < RISK_LOWER:
                self.riskFactor = RISK_LOWER
            if self.riskFactor > RISK_UPPER:
                self.riskFactor = RISK_UPPER

        if random.randint(0, 100) >= self.riskFactor * (1 - mgf) * (1 + self.price / 1000):
            x = (1 + abs(mgf) * random.randint(0, 10)/100)
            #print(f'upwards: {x}')
            self.price = self.price * x
        else:
            x = (1 - abs(mgf) * random.randint(0, 10)/100)
            #print(f'downwards:  {x}')
            self.price = self.price * x

        self.priceHistory[tick] = self.price


class investor():
    name = None
    uid = None
    money = None
    portfolio = None
    transactions = None
    history = None
    limits = None

    def __init__(self, name, uid, money=1500.0, portfolio=None, transactions=None, history=None, limits=None):
        self.name = name
        self.uid = uid
        self.money = money
        if portfolio is None:
            self.portfolio = {} # {STK: amount}
        else:
            exec(f'self.portfolio = {portfolio}')

        if transactions is None:
            self.transactions = []
        else:
            exec(f'self.transactions = {transactions}')

        if history is None:
            self.history = {}
        else:
            exec(f'self.history = {history}')

        if limits is None:
            self.limits = []
        else:
            exec(f'self.limits = {limits}')

    def update(self, tick, stockPrice):
        if tick % 1000 == 0:
            self.money += 100.0
        self.history[tick] = stockPrice + self.money

    def buy(self, stock, amount, priceeach):
        if amount.lower() == 'all':
            amount = int(self.money / priceeach)
        try:
            amount = int(amount)
        except:
            return {'error': 'That\'s not a real number.'}
        if amount <= 0:
            return {'error': 'Try buying a real amount of stocks plz'}
        total_price = priceeach * amount

        if self.money >= total_price:
            self.money -= total_price
            if stock in self.portfolio:
                self.portfolio[stock] = self.portfolio[stock] + amount
            else:
                self.portfolio[stock] = {}
                self.portfolio[stock] = amount
            self.transactions.append(f'Bought {amount} shares of {stock} for {priceeach} each.')
            return {'msg': random.choice(['Nice Purchase', 'Wrong Move, dumdum', '... Are you sure?', 'I have a good feeling about this.', 'lmao', 'To the mooooooooooooon'])}
        else:
            return {'error': 'Not enough money loserman'}

    def sell(self, stock, amount, priceeach):
        if stock in self.portfolio:
            if amount == 'all':
                amount = self.portfolio[stock]
            else:
                amount = int(amount)
            if self.portfolio[stock] >= amount:
                total_price = priceeach * amount
                self.portfolio[stock] -= amount

                self.money += total_price
                self.transactions.append(f'Sold {amount} shares of {stock} for {priceeach} each.')
                if self.portfolio[stock] == 0:
                    del self.portfolio[stock]
                return {'msg': random.choice(['Nice Sell', 'Wrong Move, dumdum', '... Are you sure?', 'I have a good feeling about this.', 'lmao', 'Wrong moon?'])}
            else:
                return {'error': 'You do not have that many shares stoopid'}
        else:
            return {'error': 'You don\'t even have that stock '}

    def limitBuy(self, stock, amount:int, limit:float):
        if self.money >= amount * limit:
            self.limits.append({'stock': stock, 'amount':amount, 'limit':limit, 'type': 'buy'})
            self.money -= amount * limit
            return {'msg': 'Limit set!'}
        else:
            return {'error': 'Not enough money loserman'}

    def limitSell(self, stock, amount, limit):
        if stock in self.portfolio:
            if amount == 'all':
                amount = self.portfolio[stock]

            if self.portfolio[stock] >= amount:
                self.limits.append({'stock': stock, 'amount':amount, 'limit':limit, 'type': 'sell'})
                return {'msg': 'Limit set!'}
            else:
                return {'error': 'You do not have that many shares stoopid'}
        else:
            return {'error': 'You don\'t even have that stock '}

queue = redis.StrictRedis(host='localhost', port=6379, db=0)
pubsub = queue.pubsub()
pubsub.subscribe('marketCommands')

def send_zipped_pickle(obj, feed):
    p = pickle.dumps(obj)
    z = zlib.compress(p)
    return queue.publish(feed, z)

def recv_zipped_pickle():
    z = pubsub.get_message()
    if z is not None:
        if isinstance(z['data'], bytes):
            p = zlib.decompress(z['data'])
            return pickle.loads(p)

async def API():
    while True:
        msg = recv_zipped_pickle()
        if msg:
            rtn = ''
            if msg['command'] == 'trends':
                rtn = await stonks.showTrends()
            elif msg['command'] == 'showMarket':
                rtn = await stonks.showMarket(msg['stocks'], msg['time'])
            elif msg['command'] == 'openAccount':
                rtn = await stonks.openAccount(msg['name'], msg['id'])
            elif msg['command'] == 'showInvestors':
                rtn = await stonks.showInvestors()
            elif msg['command'] == 'lossPorn':
                rtn = await stonks.lossPorn(msg['id'], msg['time'])
            elif msg['command'] == 'showHoldings':
                rtn = await stonks.showHoldings(msg['id'])
            elif msg['command'] == 'buy':
                rtn = await stonks.buy(msg['id'], msg['stock'], msg['amount'])
            elif msg['command'] == 'sell':
                rtn = await stonks.sell(msg['id'], msg['stock'], msg['amount'])
            elif msg['command'] == 'limitBuy':
                rtn = await stonks.limitBuy(msg['id'], msg['stock'], msg['amount'], msg['limit'])
            elif msg['command'] == 'limitSell':
                rtn = await stonks.limitSell(msg['id'], msg['stock'], msg['amount'], msg['limit'])
            elif msg['command'] == 'showLimits':
                rtn = await stonks.showLimits(msg['id'])
            elif msg['command'] == 'cancelLimit':
                rtn = await stonks.cancelLimit(msg['id'], msg['index'])
            rtn['commandID'] = msg['commandID']
            send_zipped_pickle(rtn, 'marketReturns')
        await asyncio.sleep(.5)

if __name__ == '__main__':
    stonks = stock_engine()
    loop = asyncio.get_event_loop()
    loop.create_task(stonks.startMarket())
    loop.create_task(API())
    print('done')
    loop.run_forever()

