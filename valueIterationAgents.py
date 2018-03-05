# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()

        for i in range(self.iterations):
            current_vals = self.values.copy()
            for state in states:
                if not self.mdp.isTerminal(state):
                    action = self.getAction(state)
                    current_vals[state] = self.computeQValueFromValues(state, action)
            self.values = current_vals


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q = 0
        newStates = self.mdp.getTransitionStatesAndProbs(state, action)
        for x in newStates:
            q += x[1] * (self.mdp.getReward(state, action, x[0]) + self.discount * self.values[x[0]])
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        newDict = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            newDict[action] = self.computeQValueFromValues(state, action)
        return newDict.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def runValueIteration(self):
        states = self.mdp.getStates()

        for i in range(self.iterations):
            current_vals = self.values.copy()
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                action = self.getAction(state)
                current_vals[state] = self.computeQValueFromValues(state, action)
            self.values = current_vals

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
          predecessors[state] = set()
        for state in self.mdp.getStates():
          for action in self.mdp.getPossibleActions(state):
            for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
              if prob > 0:
                currPreds = predecessors[nextState]
                currPreds.add(state)
                predecessors[nextState] = currPreds

        pq = util.PriorityQueue()

        for s in self.mdp.getStates():
          if not self.mdp.isTerminal(s):
            maxQ = float('-inf')
            for a in self.mdp.getPossibleActions(s):
              if (maxQ < self.computeQValueFromValues(s, a)):
                maxQ = self.computeQValueFromValues(s, a)
            if maxQ == float('-inf'):
              diff = self.values[s]
            else:
              diff = abs(self.values[s] - maxQ)
            pq.push(s, -diff)

        for i in range(self.iterations):
          if pq.isEmpty():
            return
          s = pq.pop()
          if not self.mdp.isTerminal(s):
            bestAction = self.computeActionFromValues(s) # TODO this line and next are sus
            self.values[s] = self.computeQValueFromValues(s, bestAction) # TODO how do I update the value of s?
          for p in predecessors[s]:
            if not self.mdp.isTerminal(p):
              maxQ = float('-inf')
              for a in self.mdp.getPossibleActions(p):
                if (maxQ < self.computeQValueFromValues(p, a)):
                  maxQ = self.computeQValueFromValues(p, a)
              if maxQ == float('-inf'):
                diff = abs(self.values[p])
              else:
                diff = abs(self.values[p] - maxQ)
              if (diff > self.theta):
                pq.update(p, -diff)
                # pq.push(p, -diff)










