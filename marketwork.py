from mesa import Agent, Model
from mesa.time import RandomActivation
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import re

# 1. GPT-2를 기반으로 한 에이전트 정의
class GPT2Agent(Agent):
    def __init__(self, unique_id, model, agent_type, memory_size=5):
        super().__init__(unique_id, model)
        self.agent_type = agent_type  # "buyer" 또는 "seller"
        self.price = None
        self.memory = []  # 과거 거래 가격을 저장할 리스트
        self.memory_size = memory_size  # 기억할 거래 수 제한
        self.match = False  # 매칭 여부
        self.active = True  # 거래 가능 여부
        self.x = random.uniform(0, 10)  # 에이전트의 x 좌표 (랜덤 초기화)
        self.y = random.uniform(0, 10)  # 에이전트의 y 좌표 (랜덤 초기화)

        # 모델과 토크나이저는 Model 클래스의 속성을 참조하도록 변경
        self.tokenizer = model.tokenizer
        self.gpt2_model = model.gpt2_model

    def remember(self, price):
        """거래 가격을 기억에 추가"""
        self.memory.append(price)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # 기억 용량 초과 시 가장 오래된 기억을 삭제

    def get_gpt2_decision(self):
        """GPT-2 모델을 사용해 가격 결정"""
        memory_string = ', '.join([str(p) for p in self.memory]) if self.memory else "No previous transactions"
        prompt = f"As a {self.agent_type}, you have participated in previous transactions with prices: {memory_string}. What should be the price for the next transaction? Provide only a numerical value."

        # GPT-2로 텍스트 생성
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.gpt2_model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 결과에서 가격 추출 (정규 표현식 사용)
        price_match = re.search(r'\d+(\.\d+)?', generated_text)
        price = float(price_match.group()) if price_match else random.uniform(50, 130)

        # GPT-2가 결정한 가격을 기억에 추가
        self.remember(price)

        return price

    def step(self):
        """GPT-2를 통해 가격 결정"""
        if not self.active:
            return  # 거래 완료 후 이탈한 에이전트는 더 이상 행동하지 않음

        if len(self.memory) > 0:
            self.price = self.get_gpt2_decision()
        else:
            # 기억이 없으면 기본 무작위 가격
            if self.agent_type == "buyer":
                self.price = random.uniform(50, 100)
            elif self.agent_type == "seller":
                self.price = random.uniform(80, 130)

    def leave_market(self):
        """거래 완료 후 에이전트가 시장에서 이탈"""
        self.active = False

# 2. 시장 모델 정의
class MarketModel(Model):
    def __init__(self, num_buyers, num_sellers, memory_size=5):
        super().__init__()  # 부모 클래스인 Model의 생성자 호출
        self.num_buyers = num_buyers
        self.num_sellers = num_sellers
        self.schedule = RandomActivation(self)
        self.matching_pairs = []  # 매칭된 구매자-판매자 쌍
        self.total_transactions = 0  # 총 거래 수
        self.transaction_prices = []  # 성사된 거래 가격 기록

        # GPT-2 모델과 토크나이저를 한 번만 로드
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

        # 구매자 및 판매자 에이전트 생성
        for i in range(self.num_buyers):
            buyer = GPT2Agent(i, self, agent_type="buyer", memory_size=memory_size)
            self.schedule.add(buyer)

        for i in range(self.num_sellers):
            seller = GPT2Agent(i + self.num_buyers, self, agent_type="seller", memory_size=memory_size)
            self.schedule.add(seller)

    def step(self):
        # 모든 에이전트가 행동
        self.schedule.step()

        # 매칭 알고리즘 (구매자와 판매자의 제시 가격이 충족되면 매칭)
        buyers = [agent for agent in self.schedule.agents if agent.agent_type == "buyer" and agent.active]
        sellers = [agent for agent in self.schedule.agents if agent.agent_type == "seller" and agent.active]

        for buyer in buyers:
            for seller in sellers:
                if not buyer.match and not seller.match and buyer.price >= seller.price:
                    # 매칭이 성사되면 두 에이전트는 매칭 상태로 설정
                    buyer.match = True
                    seller.match = True
                    transaction_price = (buyer.price + seller.price) / 2  # 거래 가격을 중간값으로 설정
                    self.transaction_prices.append(transaction_price)  # 거래 가격 기록
                    self.total_transactions += 1
                    print(f"Transaction: Buyer at {buyer.price:.2f} - Seller at {seller.price:.2f} - Price: {transaction_price:.2f}")

                    # 매칭된 가격을 에이전트의 기억에 추가
                    buyer.remember(transaction_price)
                    seller.remember(transaction_price)

                    # 거래 완료 후 두 에이전트는 시장에서 이탈
                    buyer.leave_market()
                    seller.leave_market()

        # 매칭된 에이전트 초기화 (다음 스텝을 위해)
        for agent in self.schedule.agents:
            agent.match = False

# 3. 시뮬레이션 실행 및 결과 시각화
def run_market_simulation(num_buyers, num_sellers, memory_size=5, steps=50):
    model = MarketModel(num_buyers, num_sellers, memory_size)

    # Matplotlib 설정
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # 시각화 범위 설정
    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(0, 10)
    ax[0].set_title("Agent Positions")
    ax[0].set_xlabel("X Coordinate")
    ax[0].set_ylabel("Y Coordinate")

    ax[1].set_xlim(0, steps)
    ax[1].set_ylim(50, 130)
    ax[1].set_title("Transaction Prices Over Time")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Price")

    transaction_prices = []

    for step in range(steps):
        print(f"Step {step + 1}")
        model.step()

        # 현재 에이전트들의 위치 시각화
        ax[0].clear()
        ax[0].set_xlim(0, 10)
        ax[0].set_ylim(0, 10)
        ax[0].set_title("Agent Positions")
        ax[0].set_xlabel("X Coordinate")
        ax[0].set_ylabel("Y Coordinate")

        buyers = [agent for agent in model.schedule.agents if agent.agent_type == "buyer" and agent.active]
        sellers = [agent for agent in model.schedule.agents if agent.agent_type == "seller" and agent.active]

        # 구매자와 판매자의 위치를 다른 색으로 표시
        buyer_x = [agent.x for agent in buyers]
        buyer_y = [agent.y for agent in buyers]
        seller_x = [agent.x for agent in sellers]
        seller_y = [agent.y for agent in sellers]

        ax[0].scatter(buyer_x, buyer_y, c='blue', label='Buyers', alpha=0.6)
        ax[0].scatter(seller_x, seller_y, c='red', label='Sellers', alpha=0.6)
        ax[0].legend()

        # 거래 가격 시각화
        transaction_prices.extend(model.transaction_prices)
        model.transaction_prices.clear()  # 거래 가격을 시각화한 후 초기화하여 중복 방지
        ax[1].clear()
        ax[1].set_xlim(0, steps)
        ax[1].set_ylim(50, 130)
        ax[1].set_title("Transaction Prices Over Time")
        ax[1].set_xlabel("Step")
        ax[1].set_ylabel("Price")
        ax[1].plot(range(len(transaction_prices)), transaction_prices, marker='o', color='b')

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

# 시뮬레이션 실행 (예: 10명의 구매자
run_market_simulation(50, 50, memory_size=5)
