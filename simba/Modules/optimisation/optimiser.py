import random
from functools import partial
from deap import algorithms
from deap import tools
import csv


class optimiser:

    interrupt = False

    def finish_running(self, signal, frame):
        self.interrupt = True
        print("Finishing after this generation!")

    def gaSimple(
        self,
        pop,
        toolbox,
        nSelect=None,
        CXPB=0.5,
        MUTPB=0.2,
        ngen=100,
        stats=None,
        halloffame=None,
        hoffile=None,
        verbose=__debug__,
    ):

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        print("Start of evolution")

        # Variable keeping track of the number of generations
        g = 0

        # Evaluate the entire population
        eval_func = partial(toolbox.evaluate, gen=g)
        fitnesses = list(toolbox.map(eval_func, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean**2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        if halloffame is not None:
            halloffame.update(pop)
            with open(hoffile, "w") as out:
                csv_out = csv.writer(out)
                for row in halloffame:
                    row.append(0)
                    csv_out.writerow(row)

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(pop), **record)
        if verbose:
            print(logbook.stream)

        # Begin the evolution
        while g < ngen:
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)

            # Number of individual to select
            nSelect = len(pop)
            # Select the next generation individuals
            offspring = toolbox.select(pop, nSelect)
            # print('population', pop)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            # print('Offspring before', offspring)
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            # print('Offspring after mating', offspring)

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # print('Offspring after mutation', offspring)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            eval_func = partial(toolbox.evaluate, gen=g)
            fitnesses = toolbox.map(eval_func, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean**2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
                with open(hoffile, "a") as out:
                    csv_out = csv.writer(out)
                    for row in halloffame:
                        row.append(g)
                        csv_out.writerow(row)

            # Append the current generation statistics to the logbook
            record = stats.compile(pop) if stats else {}
            logbook.record(gen=g, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    def eaSimple(
        self,
        population,
        toolbox,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        hoffile=None,
        verbose=__debug__,
    ):

        evaluationID = 0

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        ids = list(range(evaluationID, evaluationID + len(invalid_ind)))
        evaluationID += len(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)
            with open(hoffile, "w") as out:
                csv_out = csv.writer(out)
                for row in halloffame:
                    row.append(0)
                    csv_out.writerow(row)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            if self.interrupt:
                self.interrupt = False
                break
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            ids = list(range(evaluationID, evaluationID + len(invalid_ind)))
            evaluationID += len(invalid_ind)
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
                with open(hoffile, "a") as out:
                    csv_out = csv.writer(out)
                    for row in halloffame:
                        row.append(gen)
                        csv_out.writerow(row)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    def eaMuPlusLambda(
        self,
        population,
        toolbox,
        mu,
        lambda_,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        hoffile=None,
        verbose=__debug__,
    ):

        evaluationID = 0

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        ids = list(range(evaluationID, evaluationID + len(invalid_ind)))
        for ind, id in zip(invalid_ind, ids):
            ind.id = id
        evaluationID += len(invalid_ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit, id in zip(invalid_ind, fitnesses, ids):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)
            with open(hoffile, "w") as out:
                csv_out = csv.writer(out)
                for row in halloffame:
                    row.append(0)
                    row.append(row.id)
                    csv_out.writerow(row)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            ids = list(range(evaluationID, evaluationID + len(invalid_ind)))
            for ind, id in zip(invalid_ind, ids):
                ind.id = id
            evaluationID += len(invalid_ind)
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit, id in zip(invalid_ind, fitnesses, ids):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
                with open(hoffile, "a") as out:
                    csv_out = csv.writer(out)
                    for row in halloffame:
                        row.append(gen)
                        row.append(row.id)
                        csv_out.writerow(row)

            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu)

            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook
