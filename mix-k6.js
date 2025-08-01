import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const BASE_URL = 'http://10.112.154.218:30427';
const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('');
const max_user_index = 962;

let postTrend = new Trend('post_response_time');
let readTrend = new Trend('read_response_time');

export let options = {
    scenarios: {
        read_timeline_phase: {
            executor: 'ramping-arrival-rate',
            startRate: 100,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 1000,
            stages: [
                { target: 100, duration: '2m' }, // make the load steady
                { target: 100, duration: '580m' },
                { target: 300, duration: '1200m' },
                { target: 200, duration: '500m' },
                { target: 300, duration: '400m' },
                { target: 1, duration: '200m' },
                //{ target: 200, duration: '30m' },
                { target: 1, duration: '35h'},
                //{ target: 300, duration: '30m' },
                { target: 300, duration: '500m'},
                { target: 200, duration: '2h' },
            ],
            exec: 'readHomeTimeline', 
            startTime: '0s', 
            gracefulStop: '5m', // 可选：平滑停止
           // duration: '1h',      // 设置持续时间为1小时
        },
        compose_post_phase: {
            executor: 'ramping-arrival-rate',
            startRate: 100,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 1000,
            stages: [
                { target: 100, duration: '6h' },
                { target: 300, duration: '15h' },
                //{ target: 300, duration: '1m'},
                //{ target: 200, duration: '10s' },
                //{ target: 200, duration: '1m'},
                //{ target: 300, duration: '10s' },
                //{ target: 300, duration: '1m' },
                { target: 100, duration: '4h' },
                { target: 500, duration: '10h' },
            ],
            exec: 'composePost', 
            startTime: '2880m', 
            gracefulStop: '5m',
            //duration: '2h',      /
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<1000'],
        http_req_failed: ['rate<0.01'],
    },
};

function randomString(length) {
    let text = '';
    for (let i = 0; i < length; i++) {
        text += charset[Math.floor(Math.random() * charset.length)];
    }
    return text;
}

function randomDigits(length) {
    const digits = '0123456789';
    let text = '';
    for (let i = 0; i < length; i++) {
        text += digits[Math.floor(Math.random() * digits.length)];
    }
    return text;
}

export function composePost() {
    let user_index = Math.floor(Math.random() * max_user_index);
    let user_id = user_index.toString();
    let username = `username_${user_id}`;
    let text = randomString(256);

    let num_user_mentions = Math.floor(Math.random() * 6);
    let num_urls = Math.floor(Math.random() * 6);
    let num_media = Math.floor(Math.random() * 5);

    for (let i = 0; i < num_user_mentions; i++) {
        let mention_id = Math.floor(Math.random() * max_user_index);
        if (mention_id !== user_index) {
            text += ` @username_${mention_id}`;
        }
    }

    for (let i = 0; i < num_urls; i++) {
        text += ` http://${randomString(64)}`;
    }

    let media_ids = [];
    let media_types = [];

    for (let i = 0; i < num_media; i++) {
        media_ids.push(randomDigits(18));
        media_types.push("png");
    }

    let payload = `username=${username}&user_id=${user_id}&text=${encodeURIComponent(text)}&post_type=0`;

    if (num_media > 0) {
        payload += `&media_ids=[${media_ids.map(id => `"${id}"`).join(',')}]&media_types=[${media_types.map(t => `"${t}"`).join(',')}]`;
    }

    let headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    };

    let res = http.post(`${BASE_URL}/wrk2-api/post/compose`, payload, { headers });
    postTrend.add(res.timings.duration);

    check(res, {
        'composePost status is 200': (r) => r.status === 200,
    });

    sleep(0.01 + Math.random() * 0.05);
}

export function readUserTimeline() {
    let user_id = Math.floor(Math.random() * max_user_index).toString();
    let start = Math.floor(Math.random() * 100);
    let stop = start + 10;

    let url = `${BASE_URL}/wrk2-api/user-timeline/read?user_id=${user_id}&start=${start}&stop=${stop}`;

    let res = http.get(url);
    readTrend.add(res.timings.duration);

    check(res, {
        'readUserTimeline status is 200': (r) => r.status === 200,
    });

    sleep(1 + Math.random() * 5);
}

export function readHomeTimeline() {
    let user_id = Math.floor(Math.random() * max_user_index).toString();
    let start = Math.floor(Math.random() * 100);
    let stop = start + 10;

    let url = `${BASE_URL}/wrk2-api/home-timeline/read?user_id=${user_id}&start=${start}&stop=${stop}`;

    let res = http.get(url);
    readTrend.add(res.timings.duration);

    check(res, {
        'readHomeTimeline status is 200': (r) => r.status === 200,
    });

    sleep(1 + Math.random() * 5);
}

