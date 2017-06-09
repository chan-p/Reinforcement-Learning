import random
import bandit

class DSP:
    def __init__(self, num_Qaction):
        self.__num_Qaction = num_Qaction
        self.__bandit_Epsil = bandit.EpsilonGreedy(0.4, num_Qaction)
        self.__bandit_UCB   = bandit.UpperConfidenceBound(num_Qaction)

    def push_ad_random(self):
        return random.choice([i for i in range(self.__num_Qaction)])

    def reserve_reward_random(self, reward):
        return

    def push_ad_Egreedy(self):
        return self.__bandit_Epsil.get_action()

    def reserve_reward_Egreedy(self, action, reward):
        self.__bandit_Epsil.reserve_reward(action, reward)

    def push_ad_UCB(self):
        return self.__bandit_UCB.get_action()

    def reserve_reward_UCB(self, reward):
        self.__bandit_UCB.reserve_reward(reward)

class User:
    def __init__(self, context_info):
        self.__context_info = context_info
        self.context = []

    def __init_context(self):
        self.context = []

    def generate(self):
        self.__init_context()
        for context in self.__context_info:
            prob = random.randint(0, 100)
            value = 0.
            for val in context:
                value += val[1]
                if prob <= value:
                    self.context.append(val[0])
                    break

    def judge_click(self, user, ad):
        click_prob = 1.0
        for user_context in user.context:
            click_prob = click_prob * (ad[user.context.index(user_context)][user_context]/100)
        return 1 if click_prob > random.random() else 0

class Advertise:
    def __init__(self, attribute_info):
        self.__attribute_info = attribute_info
        self.attribute_pattern = []
        self.num_pattern = 1

    def generate(self):
        pattern_list = []
        for info in self.__attribute_info:
            self.num_pattern = self.num_pattern * len(info)

        for i in range(1000):
            pattern = []
            for info in self.__attribute_info: pattern.append(random.choice(info))
            if pattern not in self.attribute_pattern: self.attribute_pattern.append(pattern)
            if len(self.attribute_pattern) == self.num_pattern: break

def run():
    user_context = [[(0, 30), (1, 70)], [(0, 10), (1, 20), (2, 30), (3, 40)]]
    ad_attribute = [[(10, 2), (2, 10)], [(2, 3, 1, 1), (10, 2, 2, 5)]]
    user = User(user_context)
    ad = Advertise(ad_attribute)
    ad.generate()
    total_imp = 100000000
    dsp = DSP(ad.num_pattern)
    imp = [0 for i in range(ad.num_pattern)]
    click = [0 for i in range(ad.num_pattern)]
    for index in range(total_imp):
        user.generate()
        push_ad_number = dsp.push_ad_random()
        # push_ad_number = dsp.push_ad_Egreedy()
        # push_ad_number = dsp.push_ad_UCB()
        imp[push_ad_number] += 1
        if user.judge_click(user, ad.attribute_pattern[push_ad_number]):
            click[push_ad_number] += 1
            dsp.reserve_reward_random(1)
            dsp.reserve_reward_Egreedy(push_ad_number, 1)
            #dsp.reserve_reward_UCB(1)
        else:
            dsp.reserve_reward_random(0)
            dsp.reserve_reward_Egreedy(push_ad_number, 0)
            #dsp.reserve_reward_UCB(0)
    total_CTR = 0.
    for index in range(ad.num_pattern):
        print(ad.attribute_pattern[index], click[index]/imp[index])
        total_CTR += click[index]/imp[index]
    print(total_CTR/ad.num_pattern)

if __name__=="__main__":
    run()
