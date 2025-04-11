import time

profiler = dict()


def add_timing(tag, time):
    if tag in profiler.keys():
        profiler[tag] += time
    else:
        profiler[tag] = time


def print_timings():
    print("Overview of timings steps \n")

    print("{:40}".format("Code section"), "runtime [s]    percentage")
    print("-----------------------------------------------------------")
    total = 0.0
    for k, v in profiler.items():
        total += v

    for k, v in profiler.items():
        print("{:40}".format(k), "%3.4f" % v, "    ", "%2.3f" % (100.0 * v / total))
        # print("'%s' % 2.4f    [ %2.2f %%] " % (k, v, 100. * v/total))

    print("-----------------------------------------------------------")
    print("{:40}".format("Total"), "%3.4f" % total)
