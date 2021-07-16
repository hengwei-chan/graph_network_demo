if [-f "already_ran"]; then
    echo "Already ran the Entrypoint once. Holding indefinitely for debugging."
    /bin/sh/
fi
touch already_ran
####
-v /home/oliver/projects/try_2.xls:/data/try.xls --name logi logd_predict:1.0.0 --input /data/try.xls --output /data/out.txt > /home/oliver/output.txt