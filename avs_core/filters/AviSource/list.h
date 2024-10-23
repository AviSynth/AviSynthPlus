// Packaged with Avisynth v1.0 beta.
// http://www.math.berkeley.edu/~benrg/avisynth.html

//	VirtualDub - Video processing and capture application
//	Copyright (C) 1998-2000 Avery Lee
//
//	This program is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program; if not, write to the Free Software
//	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

#ifndef f_LIST_H
#define f_LIST_H

class ListNode {
public:
	ListNode *next, *prev;

  ListNode() {}
  ListNode(void *pv) {}

	void Remove() {
		next->prev = prev;
		prev->next = next;
	}

	void InsertAfter(ListNode *node) {
		next = node;
		prev = node->prev;
		if (node->prev) node->prev->next = this;
		node->prev = this;
	}

	void InsertBefore(ListNode *node) {
		next = node->next;
		prev = node;
		if (node->next) node->next->prev = this;
		node->next = this;
	}

	ListNode *NextFromHead() {
		return prev;
	}

	ListNode *NextFromTail() {
		return next;
	}
};

class List {
private:
public:
	ListNode head, tail;

	// <--- next             prev --->
	//
	// head <-> node <-> node <-> tail

	List();

	void AddHead(ListNode *node) {
		node->InsertAfter(&head);
	}

	void AddTail(ListNode *node) {
		node->InsertBefore(&tail);
	}

	ListNode *RemoveHead();
	ListNode *RemoveTail();

	bool IsEmpty() {
		return !head.prev->prev;
	}

	ListNode *AtHead() {
		return head.prev;
	}

	ListNode *AtTail() {
		return tail.next;
	}
};

// Templated classes... no code.

template<class T> class List2;

template<class T>
class ListNode2 : public ListNode {
friend List2<T>;
public:
	(ListNode2<T>)() {}
	(ListNode2<T>)(void *pv) : ListNode(pv) {}

	void InsertBefore(ListNode2<T> *node) { ListNode::InsertBefore(node); }
	void InsertAfter(ListNode2<T> *node) { ListNode::InsertAfter(node); }

	void Remove() { ListNode::Remove(); }
	T *NextFromHead() { return (T *)ListNode::NextFromHead(); }
	T *NextFromTail() { return (T *)ListNode::NextFromTail(); }
};

template<class T>
class List2 : public List {
public:
	(List2<T>)() {}

	void AddHead(ListNode2<T> *node) { List::AddHead(node); }
	void AddTail(ListNode2<T> *node) { List::AddTail(node); }
	T *RemoveHead() { return (T *)List::RemoveHead(); }
	T *RemoveTail() { return (T *)List::RemoveTail(); }
	T *AtHead() { return (T *)List::AtHead(); }
	T *AtTail() { return (T *)List::AtTail(); }
};

#endif
