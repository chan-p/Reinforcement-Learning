import random
import bandit

class DSP:
    def __init__(self, num_Qaction, mode):
        self.mode = mode
        self.__num_Qaction = num_Qaction
        if mode == "Random":
            self.bandit = bandit.Random(num_Qaction)
        elif mode == "EpsilonGreedy":
            self.bandit = bandit.EpsilonGreedy(0.4, num_Qaction)
        elif mode == "UpperConfidenceBound":
            self.bandit = bandit.UpperConfidenceBound(num_Qaction)
        else:
            print("Mode Error")

    def push_ad(self):
        return self.bandit.get_action()

    def reserve_reward(self, reward):
        self.bandit.reserve_reward(reward)

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

def run(user, ad, dsp, total_imp):
    print("Total_Imp:" + str(total_imp))
    imp = [0 for i in range(ad.num_pattern)]
    click = [0 for i in range(ad.num_pattern)]
    for index in range(total_imp):
        user.generate()
        push_ad_number = dsp.push_ad()
        imp[push_ad_number] += 1
        if user.judge_click(user, ad.attribute_pattern[push_ad_number]):
            click[push_ad_number] += 1
            dsp.reserve_reward(1)
        else:
            dsp.reserve_reward(0)

    total_CTR = 0.
    for index in range(ad.num_pattern):
        print(ad.attribute_pattern[index], click[index]/imp[index])
        total_CTR += click[index]/imp[index]
    print(total_CTR/ad.num_pattern)

if __name__=="__main__":
    user_context = [[(0, 30), (1, 70)], [(0, 10), (1, 20), (2, 30), (3, 40)]]
    ad_attribute = [[(10, 2), (2, 10)], [(2, 3, 1, 1), (10, 2, 2, 5)]]
    user = User(user_context)
    ad   = Advertise(ad_attribute)
    ad.generate()
    dsp = DSP(ad.num_pattern, "Random")
    # dsp = DSP(ad.num_pattern, "EpsilonGreedy")
    # dsp = DSP(ad.num_pattern, "UpperConfidenceBound")
    print("Mode:" + dsp.mode)
    run(user, ad, dsp, 1000000)
