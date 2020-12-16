from random import random


from components.go import goEngineApi


class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------

        """
        # Actual MCTS Simulations (E
        for _ in range(0, simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            #TODO schätzung für reward muss von NN kommen --> Wird hochpropagiert
            v.backpropagate(reward)
            #search_function(self.)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def search_function(self, simulations_number):
        for _ in range(0, simulations_number):
            v = self._tree_policy()

            v.backpropagate(v.winner)

        return self.root.best_child(c_param=0.)

            #TODO: Fall Terminal Node abfangen

            # if v.is_terminal_node():
            #     v.backpropagate(v.p_distr, v.q_value)
            #     # backprop actual result else predicted result
            #     return
            # else:
                #netz fragen

        # #Check if returned Node = terminal node
        #
        # # Instead of Rollout: Pass v + q up along the search path
        # # Ask neural net for board state returned by self.treepolicy
        # # Except: Is terminal state, then propagate actual result
        # reward = v.backpropagate(v.results)
        # # TODO schätzung für das beste kind muss von NN kommen
        # v.backpropagate(reward)

    def _tree_policy(self):
        """
        node selection
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        # TODO
        # fuer alphazero anpassen

        while not current_node.is_terminal_node():

            # aktuelle knoten nehmen
            # 1. prüfen ob er ncoh nicht besucht uwrde
            #     1.1 NN um hilfe fragen -> P[s][a] und v
            # 2. falls nicht -> alle kinder durchgehen "probieren"
            #     2.1 und dann returnen (? stimmt das so ?)
            # 3. neuer knoten = bestes kind
            # 4. nächste schleifeniteration

            # step 1
            if current_node.n == 0:
                #current_node.n += 1 #muesste beim backpropagaten erhohet werden
                #TODO auf reale werte des NN aendern
                current_node.winner = self.randomWinner() #von NN
                current_node.p_distr = goEngineApi.getMockProbabilities(current_node.state.pos) #von NN

                return current_node

            # step 2
            if not current_node.is_fully_expanded():

                while not current_node.is_fully_expanded():
                    current_node.expand()  # Setzen von q,n = 0

                return current_node


            # step 3 + 4
            current_node = current_node.best_child() #best_child geht den schritt in das beste kind

            # -- ALTER PART --
            # # if self.node not in self.visitedNodes:
            # #     self.visitedNodes.append(current_node)
            # #     return current_node
            # if current_node.n == 0:
            #     return current_node
            #
            # #ausprobieren der kinder
            # if not current_node.is_fully_expanded():
            #     return current_node.expand()
            # else:
            #     current_node = current_node.best_child()

        #außerhalb der while schleife
        if current_node.is_terminal_node():
            winner = current_node.state.game_result
            current_node.winner = current_node.state.game_result
            current_node.p_distr = goEngineApi.getMockProbabilities(current_node.state.pos)  # von NN


        return current_node



    def randomWinner(self):
        return 1 if random() < 0.5 else -1
        #return random.random()