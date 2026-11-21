package com.example.test;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;

public class MyNewProcessor extends BaseProcessor implements Runnable {
    private int count;
    private List<String> items;
    public static final String NAME = "processor";

    public MyNewProcessor(int c) {
        this.count = c;
        this.items = new ArrayList<>();
    }

    public int compute(int x, Map<String, Object> ctx) {
        int total = 0;
        for (int i = 0; i < x; i++) {
            if (i % 2 == 0 && count > 0) {
                total += i;
            } else if (i % 3 == 0 || ctx.containsKey("skip")) {
                total -= i;
            } else {
                try {
                    total += riskyOp(i);
                } catch (Exception e) {
                    total = 0;
                }
            }
        }
        switch (x) {
            case 1: total++; break;
            case 2: total--; break;
            case 3: total *= 2; break;
            default: break;
        }
        for (String item : items) {
            if (item != null && item.length() > 0) {
                total += item.hashCode() % 10;
            }
        }
        return total;
    }

    private int riskyOp(int i) throws Exception {
        if (i < 0) throw new Exception("negative");
        return i * i;
    }

    public void run() {
        // background worker loop
        while (count > 0) {
            count--;
        }
    }

    public int getCount() { return count; }
}
