import time
from django.http import JsonResponse, HttpResponse

num_workers = 0
records = None
t = 1
epsilon = 0.5


def init_state_server(request):
    global num_workers, t, epsilon, records
    t = 1
    epsilon = float(request.POST.get("epsilon", 0.5))
    num_workers = int(request.POST.get("num_workers", 6))
    records = [[r, 0, 0, 0, False, 1] for r in range(num_workers)]
    return HttpResponse()


def find_slowest():
    global num_workers, records
    slowest = None
    val = 0
    for r in range(num_workers):
        if records[r][2] > val:
            slowest = r
            val = records[r][2]
    return slowest


def apply_for_aggregation(request):
    r = int(request.POST.get("r"))
    k = int(request.POST.get("k"))
    c = float(request.POST.get("c"))
    te = float(request.POST.get("te"))
    global epsilon, records, t
    records[r][1] = k
    records[r][2] = c
    records[r][3] = te
    if k == 0:
        return JsonResponse({"ready": 0})
    slowest = find_slowest()
    if records[slowest][5] != t:
        return JsonResponse({"ready": 0})
    current_time = time.time()
    rest_time = records[slowest][2] - current_time + records[slowest][3]
    if r == slowest or records[slowest][4] or records[r][2] + epsilon > rest_time:
        records[r][4] = True
        return JsonResponse({"ready": 1})
    return JsonResponse({"ready": 0})


def reset_state_server(request):
    r = int(request.POST.get("r"))
    t_ = int(request.POST.get("t"))
    te = float(request.POST.get("te"))
    global num_workers, records, t
    if t != t_:
        t = t_
        for r_ in range(num_workers):
            records[r_][1] = 0
            records[r_][4] = False
    records[r][5] = t_
    records[r][3] = te
    return HttpResponse()


def home(request):
    global num_workers, records, t, epsilon
    return JsonResponse({"num_workers": num_workers, "t": t, "epsilon": epsilon, "records": records})
