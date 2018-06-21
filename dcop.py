# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:34:12 2018

@author: yossi
"""
import pandas as pd
import numpy as np
import random
import time
from matplotlib import pyplot as plt
import datetime


class DCOP_Algorithm:

    def __init__(self, num_iterations, problem):

        self.num_iterations = num_iterations
        self.problem = problem
        self.costs = []

    def solve(self):
        pass

    def get_problem(self):

        return self.problem

    def get_num_iterations(self):

        return self.get_num_iterations

    def calc_total_costs(self):

        self.costs.append(self.problem.get_total_costs())

    def random_assignment(self, nodes):

        for node in nodes:
            node.get_owner().assign_random()

    def send_messages(self, nodes):

        for node in nodes:
            node.get_owner().send_message()

    def find_best_alternative(self, nodes):

        for node in nodes:
            node.get_owner().find_best_alternative()

    def do_alternative(self, nodes):

        for node in nodes:
            node.get_owner().do_alternative()

    def show_solution_graph(self):

        plt.title(self.__class__.__name__)
        plt.plot(range(0, self.num_iterations), self.costs)
        plt.show()

    def show_problem_details(self):

        self.problem.get_graph().show_graph_details()

    def empty_mailboxs(self, nodes):

        for node in nodes:
            node.get_owner().empty_mailbox()

    def total_costs_update(self, nodes):

        for node in nodes:
            node.get_owner().get_total_costs()

    def get_costs(self):

        return self.costs


class DSA(DCOP_Algorithm):

    def __init__(self, num_iterations, problem, a_type):

        self.a_type = a_type
        super(DSA, self).__init__(num_iterations, problem)

    def solve(self):

        graph = self.problem.get_graph()
        nodes = graph.get_nodes()
        self.random_assignment(nodes)
        self.total_costs_update(nodes)

        for i in range(0, self.num_iterations):
            self.empty_mailboxs(nodes)
            self.send_messages(nodes)
            self.find_best_alternative(nodes)
            self.do_alternative(nodes)
            self.calc_total_costs()

        return self.problem.solution()

    def do_alternative(self, nodes):

        for node in nodes:
            node.get_owner().do_alternative(0.7, "dsa")

    def get_type(self):

        return self.a_type


class MGM(DCOP_Algorithm):

    def solve(self):

        graph = self.problem.get_graph()
        nodes = graph.get_nodes()
        self.random_assignment(nodes)
        self.total_costs_update(nodes)

        for i in range(0, self.num_iterations):
            self.empty_mailboxs(nodes)
            self.send_messages(nodes)
            self.find_best_alternative(nodes)
            self.empty_improvments(nodes)
            self.send_improvment(nodes)
            self.do_alternative(nodes)
            self.calc_total_costs()

        return self.problem.solution()

    def send_improvment(self, nodes):

        for node in nodes:
            node.get_owner().send_improvment()

    def do_alternative(self, nodes):

        for node in nodes:
            node.get_owner().do_alternative(type_a="mgm")

    def get_type(self):

        return self.a_type

    def empty_improvments(self, nodes):

        for node in nodes:
            node.get_owner().empty_improvments()


class MGM2(MGM):

    def solve(self):

        graph = self.problem.get_graph()
        nodes = graph.get_nodes()
        self.random_assignment(nodes)
        self.total_costs_update(nodes)

        for i in range(0, self.num_iterations):
            self.am_i_a_proposer(nodes)
            self.send_friend_request(nodes)
            self.decide_friend_request(nodes)
            self.empty_mailboxs(nodes)
            self.send_messages(nodes)
            self.find_best_alternative(nodes)
            self.empty_improvments(nodes)
            self.send_improvment(nodes)
            self.do_alternative(nodes)
            self.unfriend(nodes)
            self.calc_total_costs()

        return self.problem.solution()

    def am_i_a_proposer(self, nodes):

        for node in nodes:
            node.get_owner().be_a_proposer()

    def send_friend_request(self, nodes):

        for node in nodes:
            node.get_owner().send_friend_request()

    def decide_friend_request(self, nodes):

        for node in nodes:
            node.get_owner().decide_friend_request(0)

    def unfriend(self, nodes):

        for node in nodes:
            node.get_owner().unfriend()

    def send_improvment(self, nodes):

        for node in nodes:
            node.get_owner().send_improvment("mgm2")

    def do_alternative(self, nodes):

        for node in nodes:
            node.get_owner().do_alternative(type_a="mgm2")


class DBA(MGM):

    def solve(self):

        graph = self.problem.get_graph()
        nodes = graph.get_nodes()
        self.random_assignment(nodes)
        self.total_costs_update(nodes)

        for i in range(0, self.num_iterations):
            self.empty_mailboxs(nodes)
            self.send_messages(nodes)
            self.find_best_alternative(nodes)
            self.empty_improvments(nodes)
            self.send_improvment(nodes)
            self.do_alternative(nodes)
            self.calc_total_costs()
            self.check_for_QLO(nodes)

        return self.problem.solution()

    def check_for_QLO(self, nodes):

        for node in nodes:

            if node.get_owner().get_R_mine() <= 0:

                QLO = True

                for neighbor in node.get_owner().get_neighbors():

                    if neighbor["agent"].get_R_mine() > 0:
                        QLO = False
                        break

                if QLO:
                    self.adjust_broken_constraints(node.get_owner())

    def adjust_broken_constraints(self, agent):

        mul_const = 2

        for neighbor in agent.get_neighbors():
            table = neighbor["costs_table"]
            table = table * mul_const


class Agent:

    def __init__(self, id_n, domain_size):

        inf = 10000000000000

        self.id = id_n
        self.domain_size = domain_size
        self.assignment = None
        self.best_alternative = None
        self.total_costs = inf
        self.best_alternative_costs = inf
        self.proposer = False
        self.new_friend = None
        self.been_asked_for_friendship = False
        self.mailbox = []
        self.neighbors = []
        self.improvments = []

    def assign_random(self):

        self.assignment = random.randint(0, self.domain_size - 1)

    def assign(self, assignment):

        self.assignment = assignment

    def get_improvment(self, improvment):

        self.improvments.append(improvment)

    def get_R_mine(self):

        r = self.total_costs - self.best_alternative_costs
        return r

    def send_improvment(self, a_type=""):

        if a_type == "mgm2" and self.new_friend != None:

            diff_1 = self.total_costs - self.best_alternative_costs
            diff = diff_1 + self.new_friend.total_costs - \
                   self.new_friend.get_best_alternative_costs()

        else:

            diff = self.total_costs - self.best_alternative_costs

        for neighbor in self.neighbors:

            if (a_type == "mgm2" and neighbor["agent"] != self.new_friend) \
                    or a_type == "":
                neighbor["agent"].get_improvment(diff)

    def get_total_costs(self):

        total_costs = 0

        for neighbor in self.neighbors:
            assignment = self.assignment
            other_assignment = neighbor["agent"].get_assignment()
            costs_table = neighbor["costs_table"]

            total_costs += costs_table[other_assignment][assignment]

        self.total_costs = total_costs

        return self.total_costs

    def empty_mailbox(self):

        self.mailbox = []

    def empty_improvments(self):

        self.improvments = []

    def send_message(self):

        for neighbor in self.neighbors:
            message = Message(self, neighbor["costs_table"], self.assignment)
            neighbor["agent"].get_message(message)

    def get_message(self, message):

        self.mailbox.append(message)

    def find_best_alternative(self):
        #        total_costs = 0
        sum_of_all = pd.Series(0, index=range(0, self.domain_size))

        for message in self.mailbox:
            costs_table = message.get_costs_table()
            assignment = message.get_node_assignment()
            series = costs_table.iloc[assignment]
            sum_of_all += series

        self.best_alternative = sum_of_all.idxmin()
        self.best_alternative_costs = sum_of_all[self.best_alternative]

    def do_alternative(self, p=None, type_a=None):

        if type_a == "dsa":

            if self.best_alternative_costs <= self.total_costs \
                    and random.random() <= p:
                self.assignment = self.best_alternative

        elif type_a == "mgm":

            R_mine = self.total_costs - self.best_alternative_costs

            if R_mine > 0 and (len(self.improvments) == 0 or R_mine > max(self.improvments)):
                self.assignment = self.best_alternative

        elif type_a == "mgm2":

            if self.new_friend != None and self.proposer == False:

                return

            elif self.new_friend != None:

                R_mine_1 = self.total_costs - self.best_alternative_costs
                R_mine = R_mine_1 + self.new_friend.total_costs - \
                         self.new_friend.get_best_alternative_costs()
                improvments = self.improvments + self.new_friend.get_improvments()

            else:

                R_mine = self.total_costs - self.best_alternative_costs
                improvments = self.improvments

            if R_mine > 0 and (len(improvments) == 0 or R_mine > max(improvments)):

                self.assignment = self.best_alternative
                if self.new_friend != None:
                    self.new_friend.set_assignment_as_alternative()

    def get_improvments(self):

        return self.improvments

    def set_assignment_as_alternative(self):

        self.assignment = self.best_alternative

    def add_neighbor(self, agent, costs_table):

        self.neighbors.append({"agent": agent, "costs_table": costs_table})

    def get_costs_table(self, agent):

        for neighbor in self.neighbors:

            if neighbor == agent:
                return neighbor["costs_table"]

    def get_domain_size(self):

        return self.domain_size

    def get_neighbors(self):

        return self.neighbors

    def get_assignment(self):

        return self.assignment

    def get_id(self):

        return self.id

    def be_a_proposer(self):

        lotery = random.random()

        if lotery < 0.5:

            self.proposer = True

        else:

            self.proposer = False

    def get_friend_request(self, new_friend):

        if self.been_asked_for_friendship == False and self.proposer == False:

            self.been_asked_for_friendship = True

            for neighbor in self.neighbors:

                if new_friend == neighbor["agent"]:
                    self.new_friend = new_friend
                    return

            raise Exception("there is no neighbor like this: " + str(new_friend.get_id()))

    def send_friend_request(self):

        if self.proposer == True:

            if len(self.neighbors) > 1:

                rand_index = random.randint(0, len(self.neighbors) - 1)
                self.neighbors[rand_index]["agent"].get_friend_request(self)
            elif len(self.neighbors) == 1:
                self.neighbors[0]["agent"].get_friend_request(self)

    def decide_friend_request(self, p):

        if self.new_friend != None:

            rand = random.random()

            if rand <= p:
                self.new_friend.new_friend = None
                self.new_friend.been_asked_for_friendship = False
                self.new_friend = None
                self.been_asked_for_friendship = False

    def unfriend(self):

        self.proposer = False
        self.new_friend = None
        self.been_asked_for_friendship = False

    def get_best_alternative_costs(self):

        return self.best_alternative_costs


class Node:

    def __init__(self, owner):
        self.owner = owner
        self.connections = []

    def get_owner(self):
        return self.owner

    def add_connection(self, node, costs_table):
        self.connections.append({"node": Node, "costs_table": costs_table})

    def remove_connection(self, node):
        self.nodes.remove(node)

    def get_connections(self):
        return self.connections

    def get_domain_size(self):
        return self.owner.get_domain_size()


class Problem():

    def __init__(self, num_of_agents, p1, p2, domain_size):

        self.num_of_agents = num_of_agents
        self.p1 = p1
        self.p2 = p2
        self.graph = self.generate_graph(domain_size)
        self.final_solution = None

    def generate_graph(self, domain_size):

        nodes = []

        for i in range(0, self.num_of_agents):
            agent = Agent(i + 1, domain_size)
            node = Node(agent)
            nodes.append(node)

        for i in range(0, self.num_of_agents):

            for j in range(i + 1, self.num_of_agents):

                if random.random() <= self.p1:

                    node_i = nodes[i]
                    node_j = nodes[j]
                    agent_i = nodes[i].get_owner()
                    agent_j = nodes[j].get_owner()
                    i_domain_size = node_i.get_domain_size()
                    j_domain_size = node_j.get_domain_size()
                    data = [[random.randint(1, 10) \
                                 if random.random() < self.p2 else 0 \
                             for i in range(0, j_domain_size)] \
                            for j in range(0, i_domain_size)]

                    if np.sum(data) > 0:
                        costs_table = pd.DataFrame(data, columns= \
                            range(0, j_domain_size))
                        node_i.add_connection(node_j, costs_table)
                        node_j.add_connection(node_i, costs_table.T)
                        agent_i.add_neighbor(agent_j, costs_table)
                        agent_j.add_neighbor(agent_i, costs_table.T)

        return Graph(nodes)

    def get_graph(self):

        return self.graph

    def get_total_costs(self):

        total_costs = 0

        for node in self.graph.get_nodes():
            total_costs += node.get_owner().get_total_costs()

        return total_costs

    def solution(self):

        total_costs = 0
        assignments = []

        for node in self.graph.get_nodes():
            total_costs += node.get_owner().get_total_costs()
            assignments.append({"node_index": node.get_owner().get_id(), \
                                "assignment": node.get_owner().get_assignment()})

        self.final_solution = {"total_costs": total_costs, "assignments": assignments}

        return self.final_solution


class Graph:

    def __init__(self, nodes):
        self.nodes = nodes

    def get_nodes(self):
        return self.nodes

    def add_node(self, node):
        self.nodes.append(node)

    def show_graph_details(self):
        for node in self.nodes:
            agent = node.get_owner()
            print("agent id: " + str(agent.get_id()))
            print("neighbores: " + str(agent.get_neighbors()))
            print("assignment: " + str(agent.assignment))
            print("total costs: " + str(agent.total_costs))
            print("total alterantive costs: " + str(agent.best_alternative_costs))
            print("best alternative: " + str(agent.best_alternative))


class Message:

    def __init__(self, agent, costs_table, assignment):
        self.agent = agent
        self.costs_table = costs_table
        self.assignment = assignment
        self.time = time.time()

    def get_agent(self):
        return self.agent

    def get_costs_table(self):
        return self.costs_table

    def get_node_assignment(self):
        return self.assignment

    def get_time(self):
        return self.time


def main():
    num_of_problems = 10
    num_of_agents = 30
    domain_size = 10
    iterations = 1000
    step = 10

    dsa_costs = []
    mgm_costs = []
    dba_costs = []
    mgm2_costs = []
    dsa_solution = []
    mgm_solution = []
    dba_solution = []
    mgm2_solution = []
    dsa_solution_mean = []
    mgm_solution_mean = []
    dba_solution_mean = []
    mgm2_solution_mean = []

    P1 = [0.5]
    P2 = np.linspace(0.1, 0.9, num_of_problems - 1)

    for p1 in P1:

        print("p1 = " + str(p1) + " the hour is: {}".format(datetime.datetime.now().time()))

        for i in range(0, num_of_problems):

            print("iteration number: {} out of {}".format(i + 1, num_of_problems) + \
                  " the hour is: {}".format(datetime.datetime.now().time()))

            for p2 in P2:
                print("p2 = " + str(p2) + \
                      " the hour is: {}".format(datetime.datetime.now().time()))
                print("before the problem was created" + \
                      " the hour is: {}".format(datetime.datetime.now().time()))

                problem = Problem(num_of_agents, p1, p2, domain_size)

                print("after the problem was created" + \
                      " the hour is: {}".format(datetime.datetime.now().time()))
                print("Starting DSA" + \
                      " the hour is: {}".format(datetime.datetime.now().time()))

                dsa = DSA(iterations, problem, "c")
                dsa_solution.append(dsa.solve()["total_costs"])

                print("Starting MGM" + \
                      " the hour is: {}".format(datetime.datetime.now().time()))

                mgm = MGM(iterations, problem)
                mgm_solution.append(mgm.solve()["total_costs"])

                print("Starting DBA" + \
                      " the hour is: {}".format(datetime.datetime.now().time()))

                dba = DBA(iterations, problem)
                dba_solution.append(dba.solve()["total_costs"])

                print("Starting MGM2" + \
                      " the hour is: {}".format(datetime.datetime.now().time()))

                mgm2 = MGM2(iterations, problem)
                mgm2_solution.append(mgm2.solve()["total_costs"])

            dsa_solution_mean.append(dsa_solution)
            mgm_solution_mean.append(mgm_solution)
            dba_solution_mean.append(dba_solution)
            mgm2_solution_mean.append(mgm2_solution)

            dsa_solution = []
            mgm_solution = []
            dba_solution = []
            mgm2_solution = []

        ("finished iterations first graphs the hour is: {}". \
         format(datetime.datetime.now().time()))

        dsa_solution_mean_df = pd.DataFrame(dsa_solution_mean).mean()
        mgm_solution_mean_df = pd.DataFrame(mgm_solution_mean).mean()
        dba_solution_mean_df = pd.DataFrame(dba_solution_mean).mean()
        mgm2_solution_mean_df = pd.DataFrame(mgm2_solution_mean).mean()

        plt.title("P1={} sum of costs".format(p1))
        plt.plot(P2, dsa_solution_mean_df, label="DSA-C=0.7")
        plt.plot(P2, mgm_solution_mean_df, label="MGM")
        plt.plot(P2, dba_solution_mean_df, label="DBA")
        plt.plot(P2, mgm2_solution_mean_df, label="MGM-2")
        plt.legend(loc=2)
        plt.show()

        print("the other graphs starting P2=1" + \
              " the hour is: {}".format(datetime.datetime.now().time()))

        for i in range(0, num_of_problems):
            print("the iteration number is: {} out of {}".format(i + 1, num_of_problems) + \
                  " the hour is: {}".format(datetime.datetime.now().time()))
            print("the problem is starting to create" + \
                  " the hour is: {}".format(datetime.datetime.now().time()))

            problem = Problem(num_of_agents, p1, 1, domain_size)

            print("the problem is finished to create" + \
                  " the hour is: {}".format(datetime.datetime.now().time()))

            print("DSA is starting" + \
                  " the hour is: {}".format(datetime.datetime.now().time()))

            dsa = DSA(iterations, problem, "c")
            dsa.solve()
            dsa_costs.append(dsa.get_costs())

            print("MGM is starting" + \
                  " the hour is: {}".format(datetime.datetime.now().time()))

            mgm = MGM(iterations, problem)
            mgm.solve()
            mgm_costs.append(mgm.get_costs())

            print("DBA is starting" + \
                  " the hour is: {}".format(datetime.datetime.now().time()))

            dba = DBA(iterations, problem)
            dba.solve()
            dba_costs.append(dba.get_costs())

            print("MGM2 is starting" + \
                  " the hour is: {}".format(datetime.datetime.now().time()))

            mgm2 = MGM2(iterations, problem)
            mgm2.solve()
            mgm2_costs.append(mgm2.get_costs())

        ("finished iterations second graphs" + \
         " the hour is: {}".format(datetime.datetime.now().time()))

        iterations_g = np.arange(0, 1000, step)

        dsa_mean = list(pd.DataFrame(dsa_costs).mean())[0::step]
        mgm_mean = list(pd.DataFrame(mgm_costs).mean())[0::step]
        dba_mean = list(pd.DataFrame(dba_costs).mean())[0::step]
        mgm2_mean = list(pd.DataFrame(mgm2_costs).mean())[0::step]

        plt.title("P1={}, P2=1 sum of costs over iterations (10 step)".format(p1))
        plt.plot(iterations_g, dsa_mean, label="DSA-C=0.7")
        plt.plot(iterations_g, mgm_mean, label="MGM")
        plt.plot(iterations_g, dba_mean, label="DBA")
        plt.plot(iterations_g, mgm2_mean, label="MGM-2")
        plt.legend(loc=1)
        plt.show()

        iterations_g = np.arange(0, 1000, 1)

        dsa_mean = list(pd.DataFrame(dsa_costs).mean())
        mgm_mean = list(pd.DataFrame(mgm_costs).mean())
        dba_mean = list(pd.DataFrame(dba_costs).mean())
        mgm2_mean = list(pd.DataFrame(mgm2_costs).mean())

        plt.title("P1={}, P2=1 sum of costs over iterations (1 step)".format(p1))
        plt.plot(iterations_g, dsa_mean, label="DSA-C=0.7")
        plt.plot(iterations_g, mgm_mean, label="MGM")
        plt.plot(iterations_g, dba_mean, label="DBA")
        plt.plot(iterations_g, mgm2_mean, label="MGM-2")
        plt.legend(loc=1)
        plt.show()

        dsa_costs = []
        mgm_costs = []
        dba_costs = []
        mgm2_costs = []


if __name__ == "__main__":
    main()