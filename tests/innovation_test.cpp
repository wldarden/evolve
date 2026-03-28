#include <evolve/innovation.h>
#include <gtest/gtest.h>

namespace ev = evolve;

TEST(InnovationCounterTest, AssignsSequentialNumbers) {
    ev::InnovationCounter counter;
    EXPECT_EQ(counter.get_or_create(0, 2), 0);
    EXPECT_EQ(counter.get_or_create(1, 2), 1);
    EXPECT_EQ(counter.get_or_create(0, 3), 2);
}

TEST(InnovationCounterTest, SamePairSameGeneration_ReturnsSameNumber) {
    ev::InnovationCounter counter;
    auto first = counter.get_or_create(0, 2);
    auto second = counter.get_or_create(0, 2);
    EXPECT_EQ(first, second);
}

TEST(InnovationCounterTest, NewGeneration_ResetsTracking) {
    ev::InnovationCounter counter;
    auto gen1 = counter.get_or_create(0, 2);
    EXPECT_EQ(gen1, 0);
    counter.new_generation();
    auto gen2 = counter.get_or_create(0, 2);
    EXPECT_EQ(gen2, 1);
}

TEST(InnovationCounterTest, CounterPersistsAcrossGenerations) {
    ev::InnovationCounter counter;
    counter.get_or_create(0, 1);
    counter.get_or_create(0, 2);
    counter.new_generation();
    auto next = counter.get_or_create(5, 6);
    EXPECT_EQ(next, 2);
}
