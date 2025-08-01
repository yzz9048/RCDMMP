import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';

const BASE_URL = 'http://10.112.154.218:32312'; // 请根据实际情况修改
const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('');
const max_user_index = 1000;

let reviewTrend = new Trend('review_response_time');

export let options = {
    scenarios: {
        compose_review_phase: {
            executor: 'ramping-arrival-rate',
            startRate: 50,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 1000,
            stages: [
                { target: 100, duration: '1m' },
                { target: 100, duration: '4h' },
                { target: 300, duration: '4h' },
                { target: 100, duration: '4h' },
                { target: 100, duration: '4h' },
                { target: 50, duration: '2h' },
                { target: 50, duration: '4h' },
                { target: 200, duration: '3h' },
                { target: 200, duration: '2h' },
                { target: 500, duration: '3h' },
                { target: 100, duration: '2h' },
                { target: 100, duration: '4h' },
                { target: 300, duration: '2h' },
                { target: 300, duration: '3h' },
                { target: 50, duration: '1h' },
                { target: 200, duration: '2h' },
                { target: 200, duration: '4h' },
            ],
            exec: 'composeReview',
            startTime: '0s',
            gracefulStop: '2m',
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<1000'],
        http_req_failed: ['rate<0.01'],
    },
};

const movie_titles = [
    "Avengers: Endgame",
    "Captain Marvel",
    "Hellboy",
    "Black Panther",
    "The Lord of the Rings: The Fellowship of the Ring",
    "Joker",
    "Titanic",
    "The Matrix",
    "Interstellar",
    "Spider-Man: Far from Home"
];

function randomString(length) {
    let result = '';
    for (let i = 0; i < length; i++) {
        result += charset[Math.floor(Math.random() * charset.length)];
    }
    return result;
}

function urlEncode(str) {
    return encodeURIComponent(str).replace(/%20/g, '+');
}

export function composeReview() {
    const movie_index = Math.floor(Math.random() * movie_titles.length);
    const user_index = Math.floor(Math.random() * max_user_index);

    const username = `username_${user_index}`;
    const password = `password_${user_index}`;
    const title = urlEncode(movie_titles[movie_index]);
    const rating = Math.floor(Math.random() * 11); // 0~10
    const text = randomString(256);

    const payload = `username=${username}&password=${password}&title=${title}&rating=${rating}&text=${text}`;
    const headers = { 'Content-Type': 'application/x-www-form-urlencoded' };

    const res = http.post(`${BASE_URL}/wrk2-api/review/compose`, payload, { headers });

    reviewTrend.add(res.timings.duration);

    check(res, {
        'composeReview status is 200': r => r.status === 200,
    });

    sleep(0.05 + Math.random() * 0.05);
}
