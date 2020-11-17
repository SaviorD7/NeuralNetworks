# [Simulation version] Run paddle fl training 

1) Make data federated by `python divider_data.py  number_of_files`. This file divides `car.data` into `number_of_files` files. After processing data's parts will be in `data` path.
You also can make your own data divider or divide your data manually.
2) As soon as your data is ready you can start your federated training by `sh run.sh`.
* FL Job Config will be generated in `fl_job_config`
* Logs will be generated in `logs`. There you can see results of the training

> You can use other stratagies in paddle fl by configure it in master. Line: `build_strategy.fed_avg = True`, but you also need confugre some parameters.
> Now using default Fed Avg stratagy, other configurations of strategies will be added ...

# [Federated version] Run paddle fl training 

1) Configure your End Points in trainer, server and scheduler.
2) Make you FL Job by starting `fl_master.py`
3) Send `fl_job_config/trainer$` to each trainer (the same for server)
4) Run `trainer.py` in each node.


## [HELP]
If you need some help you can write me:
* telegram: [@saviord](t.me/saviord7)
* vk: [@saviord7](https://vk.com/saviord7)
